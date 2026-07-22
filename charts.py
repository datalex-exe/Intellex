import io
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from flask import send_file

from database import get_table_as_df

def _empty_chart_figure(message):
    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    ax.axis('off')
    ax.text(0.5, 0.54, message, ha='center', va='center', fontsize=14, color='#94a3b8', fontweight='semibold')
    ax.text(0.5, 0.42, 'Run analysis to populate this visualization.', ha='center', va='center', fontsize=10, color='#64748b')
    fig.tight_layout()
    return fig

def _chart_response(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=160, bbox_inches='tight', facecolor='#1e293b')
    plt.close(fig)
    buffer.seek(0)
    response = send_file(buffer, mimetype='image/png', max_age=0)
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

def _currency_formatter(value, _):
    if abs(value) >= 1000000:
        return f'${value / 1000000:.1f}M'
    elif abs(value) >= 1000:
        return f'${value / 1000:.0f}k'
    return f'${value:,.0f}'

def _format_currency_text(value):
    if abs(value) >= 1000000:
        return f'${value / 1000000:,.2f}M'
    elif abs(value) >= 1000:
        return f'${value / 1000:,.1f}k'
    return f'${value:,.0f}'

def _apply_chart_style(ax, title, subtitle=None):
    ax.set_facecolor('#1e293b')
    ax.set_title(title, fontsize=16, fontweight='bold', color='#f8fafc', loc='left', pad=18)
    if subtitle:
        ax.text(0.0, 1.02, transform=ax.transAxes, s=subtitle, fontsize=9.5, color='#94a3b8')
    ax.grid(axis='y', alpha=0.15, linestyle='--', linewidth=0.8, color='#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94a3b8', labelsize=10)

def _load_sales_frame(org_id):
    df = get_table_as_df('sales_clean', org_id)
    if len(df) == 0:
        return df
    if 'order_date' in df.columns:
        df = df.copy()
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
    return df

def _get_headers_or_default(org_id):
    from database import get_column_mappings
    mappings = get_column_mappings(org_id)
    default_mappings = {
        'order_date': 'Date',
        'region': 'Region',
        'product': 'Product',
        'revenue': 'Revenue',
        'units_sold': 'Units',
        'customer_id': 'Customer'
    }
    result = {}
    for k, default_val in default_mappings.items():
        val = mappings.get(k)
        if val is None or val == '' or val.startswith('category_'):
            result[k] = default_val
        else:
            result[k] = val.replace('_', ' ').strip().title()
    return result

def _get_period_freq(df, date_col='order_date'):
    date_min = df[date_col].min()
    date_max = df[date_col].max()
    days_span = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else 0
    if days_span <= 30:
        return 'D', 'Day'
    elif days_span <= 180:
        return 'W', 'Week'
    elif days_span <= 365:
        return 'M', 'Month'
    else:
        return 'Q', 'Quarter'

def _clean_group_categories(df, cat_col, metric_col, max_categories=5):
    """
    Aggregates high-cardinality or continuous decimal columns into a clean 4-5 bar summary.
    Bins continuous numeric floats into clean range brackets (e.g. 20 - 35, 35 - 50)
    and collapses high-cardinality discrete categories into Top 4 + 'Other'.
    """
    valid_df = df.dropna(subset=[cat_col, metric_col]).copy()
    if len(valid_df) == 0:
        return pd.DataFrame(columns=[cat_col, metric_col])
        
    s = valid_df[cat_col]
    numeric_s = pd.to_numeric(s, errors='coerce')
    is_numeric = numeric_s.notna().mean() > 0.8 and s.nunique() > 8
    
    if is_numeric:
        try:
            num_series = numeric_s.dropna()
            num_bins = min(4, num_series.nunique())
            if num_bins > 1:
                bin_res, bin_edges = pd.qcut(num_series, q=num_bins, retbins=True, duplicates='drop')
                labels = [f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
                valid_df['clean_cat'] = pd.cut(numeric_s, bins=bin_edges, labels=labels, include_lowest=True).astype(str)
            else:
                valid_df['clean_cat'] = s.astype(str)
        except Exception:
            valid_df['clean_cat'] = s.astype(str)
    else:
        valid_df['clean_cat'] = s.astype(str)

    grouped = valid_df.groupby('clean_cat')[metric_col].sum().sort_values(ascending=False).reset_index()
    grouped.columns = [cat_col, metric_col]
    
    if len(grouped) > max_categories:
        top_df = grouped.iloc[:max_categories - 1].copy()
        other_sum = grouped.iloc[max_categories - 1:][metric_col].sum()
        other_row = pd.DataFrame([{cat_col: 'Other', metric_col: other_sum}])
        grouped = pd.concat([top_df, other_row], ignore_index=True)
        
    return grouped

def build_revenue_trend_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0 or 'revenue' not in df.columns or df['revenue'].dropna().empty:
        return _empty_chart_figure('No revenue data available yet.')

    headers = _get_headers_or_default(org_id)
    rev = headers['revenue']

    freq, label = _get_period_freq(df)

    trend = df.dropna(subset=['order_date', 'revenue']).copy()
    if len(trend) == 0:
        return _empty_chart_figure('No revenue data available yet.')
        
    trend['period'] = trend['order_date'].dt.to_period(freq).astype(str)
    grouped = trend.groupby('period')['revenue'].sum().reset_index()
    
    # Limit to 10 periods max for uncluttered line chart display
    if len(grouped) > 10:
        grouped = grouped.tail(10)
        
    x = list(range(len(grouped)))
    y = grouped['revenue'].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor='#1e293b')
    _apply_chart_style(ax, f'{rev} Trend by {label}', f'{label}ly {rev.lower()} momentum (Latest periods)')
    ax.plot(x, y, color='#667eea', linewidth=3.5, marker='o', markersize=8, markerfacecolor='#00d9ff', markeredgewidth=2)
    ax.fill_between(x, y, color='#667eea', alpha=0.15)
    ax.scatter([x[-1]], [y[-1]], s=180, color='#00d9ff', edgecolor='#1e293b', linewidth=2.5, zorder=4)
    ax.annotate(_format_currency_text(y[-1]), xy=(x[-1], y[-1]), xytext=(10, 10), textcoords='offset points', fontsize=10, color='#f8fafc', fontweight='bold')
    ax.set_ylabel(rev, color='#94a3b8')
    ax.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['period'], rotation=15, ha='right')
    ax.margins(x=0.05)
    fig.tight_layout()
    return fig

def build_revenue_by_region_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0 or 'region' not in df.columns or 'revenue' not in df.columns:
        return _empty_chart_figure('No regional revenue data available yet.')

    headers = _get_headers_or_default(org_id)
    rev = headers['revenue']
    reg = headers['region']

    grouped = _clean_group_categories(df, 'region', 'revenue', max_categories=5)
    if len(grouped) == 0:
        return _empty_chart_figure('No regional revenue data available yet.')

    colors = ['#667eea', '#00d9ff', '#764ba2', '#4ecdc4', '#f093fb']

    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor='#1e293b')
    _apply_chart_style(ax, f'{rev} by {reg}', f'Top 5 categories by total {rev.lower()}')
    bars = ax.barh(grouped['region'].astype(str), grouped['revenue'], color=colors[:len(grouped)], edgecolor='none', height=0.5)
    ax.invert_yaxis()
    ax.set_xlabel(rev, color='#94a3b8')
    ax.xaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    max_val = max(grouped['revenue']) if len(grouped['revenue']) > 0 else 1
    for bar, value in zip(bars, grouped['revenue'].tolist()):
        ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2, _format_currency_text(value), va='center', ha='left', fontsize=10, color='#e2e8f0', fontweight='bold')
    ax.margins(x=0.1)
    fig.tight_layout()
    return fig

def build_segment_distribution_chart(org_id):
    df = get_table_as_df('customer_segments', org_id)
    if len(df) == 0 or 'segment' not in df.columns:
        return _empty_chart_figure('No customer segment data available yet.')

    headers = _get_headers_or_default(org_id)
    cust = headers['customer_id']

    valid_df = df.dropna(subset=['segment'])
    if len(valid_df) == 0:
        return _empty_chart_figure('No customer segment data available yet.')

    grouped = valid_df['segment'].value_counts()
    if len(grouped) > 4:
        top_g = grouped.iloc[:4]
        other_sum = grouped.iloc[4:].sum()
        top_g['Other'] = other_sum
        grouped = top_g

    colors = ['#00d9ff', '#667eea', '#f093fb', '#4ecdc4', '#fce38a']
    total = grouped.sum() or 1

    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    wedges, texts, autotexts = ax.pie(
        grouped.values,
        labels=grouped.index,
        autopct=lambda pct: f'{pct:.0f}%' if pct >= 5 else '',
        startangle=90,
        colors=colors[:len(grouped)],
        wedgeprops={'width': 0.4, 'edgecolor': '#1e293b', 'linewidth': 2.5},
        pctdistance=0.75,
        textprops={'color': '#f8fafc', 'fontsize': 11, 'fontweight': 'bold'}
    )
    for text in texts:
        text.set_color('#94a3b8')
        text.set_fontsize(11)
    ax.text(0, 0.08, f'{int(total)}', ha='center', va='center', fontsize=26, fontweight='bold', color='#f8fafc')
    ax.text(0, -0.12, f'{cust}s', ha='center', va='center', fontsize=10, color='#94a3b8')
    ax.legend(wedges, grouped.index, title=f'{cust} Segments', loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    
    legend = ax.get_legend()
    if legend:
        legend.get_title().set_color('#f8fafc')
        for text in legend.get_texts():
            text.set_color('#94a3b8')

    ax.set_title(f'{cust} Segments', fontsize=16, fontweight='bold', color='#f8fafc', loc='left', pad=18)
    fig.tight_layout()
    return fig

def build_forecast_chart(org_id):
    df = get_table_as_df('forecasts', org_id)
    if len(df) == 0 or 'region' not in df.columns:
        return _empty_chart_figure('No forecast data available yet.')

    headers = _get_headers_or_default(org_id)
    rev = headers['revenue']
    reg = headers['region']

    grouped = df[['region', 'predicted_revenue', 'lower_bound', 'upper_bound']].copy().dropna(subset=['predicted_revenue'])
    if len(grouped) == 0:
        return _empty_chart_figure('No forecast data available yet.')

    if len(grouped) > 5:
        grouped = grouped.head(5)

    x = list(range(len(grouped)))
    predicted = grouped['predicted_revenue'].tolist()
    lower = grouped['lower_bound'].tolist()
    upper = grouped['upper_bound'].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor='#1e293b')
    _apply_chart_style(ax, f'{rev} Forecasts', f'Top 5 predicted {rev.lower()} trajectories')
    ax.vlines(x, lower, upper, color='#9db4ff', linewidth=10, alpha=0.35, zorder=1)
    ax.scatter(x, predicted, s=140, color='#00d9ff', edgecolor='#1e293b', linewidth=2.5, zorder=3, label='Predicted')
    ax.plot(x, predicted, color='#667eea', linewidth=2.5, zorder=2)
    max_upper = max(upper) if len(upper) > 0 else 1
    for index, value in enumerate(predicted):
        ax.text(index, value + max_upper * 0.02, _format_currency_text(value), ha='center', va='bottom', fontsize=9.5, color='#f8fafc', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['region'].astype(str), rotation=15, ha='right')
    ax.set_ylabel(rev, color='#94a3b8')
    ax.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    ax.legend(frameon=False, loc='upper left')
    
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color('#94a3b8')

    ax.margins(x=0.08)
    fig.tight_layout()
    return fig

def build_revenue_by_product_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0 or 'product' not in df.columns or 'revenue' not in df.columns:
        return _empty_chart_figure('No product revenue data available yet.')

    headers = _get_headers_or_default(org_id)
    rev = headers['revenue']
    prod = headers['product']

    grouped = _clean_group_categories(df, 'product', 'revenue', max_categories=5)
    if len(grouped) == 0:
        return _empty_chart_figure('No product revenue data available yet.')

    colors = ['#764ba2', '#667eea', '#00d9ff', '#4ecdc4', '#f093fb']

    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor='#1e293b')
    _apply_chart_style(ax, f'{rev} by {prod}', f'Top 5 categories by total {rev.lower()}')
    bars = ax.barh(grouped['product'].astype(str), grouped['revenue'], color=colors[:len(grouped)], edgecolor='none', height=0.5)
    ax.invert_yaxis()
    ax.set_xlabel(rev, color='#94a3b8')
    ax.xaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    max_val = max(grouped['revenue']) if len(grouped['revenue']) > 0 else 1
    for bar, value in zip(bars, grouped['revenue'].tolist()):
        ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2, _format_currency_text(value), va='center', ha='left', fontsize=10, color='#e2e8f0', fontweight='bold')
    ax.margins(x=0.1)
    fig.tight_layout()
    return fig

def build_units_by_region_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0 or 'region' not in df.columns or 'units_sold' not in df.columns:
        return _empty_chart_figure('No regional units data available yet.')

    headers = _get_headers_or_default(org_id)
    units = headers['units_sold']
    reg = headers['region']

    grouped = _clean_group_categories(df, 'region', 'units_sold', max_categories=5)
    if len(grouped) == 0:
        return _empty_chart_figure('No regional units data available yet.')

    colors = ['#4ecdc4', '#ff6b6b', '#667eea', '#00d9ff', '#764ba2']

    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor='#1e293b')
    _apply_chart_style(ax, f'{units} by {reg}', f'Top 5 categories by {units.lower()} sold')
    x = list(range(len(grouped)))
    bars = ax.bar(x, grouped['units_sold'], color=colors[:len(grouped)], edgecolor='none', width=0.45)
    ax.set_ylabel(units, color='#94a3b8')
    max_val = max(grouped['units_sold']) if len(grouped['units_sold']) > 0 else 1
    for bar, value in zip(bars, grouped['units_sold'].tolist()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02, f"{int(value):,}", va='bottom', ha='center', fontsize=9.5, color='#e2e8f0', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['region'].astype(str), rotation=15, ha='right')
    ax.margins(y=0.1)
    fig.tight_layout()
    return fig

def get_chart_response(chart_name, org_id):
    chart_builders = {
        'revenue-trend': lambda: build_revenue_trend_chart(org_id),
        'revenue-by-region': lambda: build_revenue_by_region_chart(org_id),
        'segment-distribution': lambda: build_segment_distribution_chart(org_id),
        'forecasts': lambda: build_forecast_chart(org_id),
        'revenue-by-product': lambda: build_revenue_by_product_chart(org_id),
        'units-by-region': lambda: build_units_by_region_chart(org_id),
    }

    builder = chart_builders.get(chart_name)
    if not builder:
        from flask import jsonify
        return jsonify({'status': 'error', 'message': 'Unknown chart'}), 404

    try:
        return _chart_response(builder())
    except Exception as e:
        print(f'Chart generation failed for {chart_name}: {e}')
        return _chart_response(_empty_chart_figure('Chart unavailable. Please run analysis again.'))
