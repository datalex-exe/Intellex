import io
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from flask import send_file

from database import get_table_as_df

def _empty_chart_figure(message):
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
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
    if abs(value) >= 1000:
        return f'${value / 1000:.0f}k'
    return f'${value:,.0f}'

def _format_currency_text(value):
    if abs(value) >= 1000:
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
    ax.tick_params(colors='#94a3b8')

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

def build_revenue_trend_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0:
        return _empty_chart_figure('No revenue data available yet.')

    headers = _get_headers_or_default(org_id)
    rev = headers['revenue']

    freq, label = _get_period_freq(df)

    trend = df.copy()
    trend['period'] = trend['order_date'].dt.to_period(freq).astype(str)
    grouped = trend.groupby('period')['revenue'].sum().reset_index()
    x = list(range(len(grouped)))
    y = grouped['revenue'].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    _apply_chart_style(ax, f'{rev} Trend by {label}', f'{label}ly {rev.lower()} momentum with latest period highlighted')
    ax.plot(x, y, color='#667eea', linewidth=3, marker='o', markersize=7, markerfacecolor='#1e293b', markeredgewidth=2)
    ax.fill_between(x, y, color='#667eea', alpha=0.12)
    ax.scatter([x[-1]], [y[-1]], s=160, color='#00d9ff', edgecolor='#1e293b', linewidth=2, zorder=4)
    ax.annotate(_format_currency_text(y[-1]), xy=(x[-1], y[-1]), xytext=(10, 10), textcoords='offset points', fontsize=10, color='#f8fafc', fontweight='bold')
    ax.set_ylabel(rev, color='#94a3b8')
    ax.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['period'], rotation=35, ha='right')
    ax.margins(x=0.04)
    fig.tight_layout()
    return fig

def build_revenue_by_region_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0 or 'region' not in df.columns:
        return _empty_chart_figure('No regional revenue data available yet.')

    headers = _get_headers_or_default(org_id)
    rev = headers['revenue']
    reg = headers['region']

    grouped = df.groupby('region')['revenue'].sum().sort_values(ascending=False).reset_index()
    colors = ['#667eea', '#00d9ff', '#764ba2', '#4ecdc4', '#f093fb', '#ff6b6b']

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    _apply_chart_style(ax, f'{rev} by {reg}', f'Ranked by total {rev.lower()} contribution')
    bars = ax.barh(grouped['region'], grouped['revenue'], color=colors[:len(grouped)], edgecolor='none', height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel(rev, color='#94a3b8')
    ax.xaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    for bar, value in zip(bars, grouped['revenue'].tolist()):
        ax.text(bar.get_width() + max(grouped['revenue']) * 0.015, bar.get_y() + bar.get_height() / 2, _format_currency_text(value), va='center', ha='left', fontsize=9, color='#e2e8f0', fontweight='bold')
    ax.margins(x=0.02)
    fig.tight_layout()
    return fig

def build_segment_distribution_chart(org_id):
    df = get_table_as_df('customer_segments', org_id)
    if len(df) == 0 or 'segment' not in df.columns:
        return _empty_chart_figure('No customer segment data available yet.')

    headers = _get_headers_or_default(org_id)
    cust = headers['customer_id']

    grouped = df['segment'].value_counts()
    colors = ['#00d9ff', '#667eea', '#f093fb', '#ff6b6b', '#4ecdc4']
    total = grouped.sum() or 1

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    wedges, texts, autotexts = ax.pie(
        grouped.values,
        labels=grouped.index,
        autopct=lambda pct: f'{pct:.0f}%' if pct >= 4 else '',
        startangle=90,
        colors=colors[:len(grouped)],
        wedgeprops={'width': 0.38, 'edgecolor': '#1e293b', 'linewidth': 2},
        pctdistance=0.8,
        textprops={'color': '#f8fafc', 'fontsize': 10, 'fontweight': 'bold'}
    )
    for text in texts:
        text.set_color('#94a3b8')
        text.set_fontsize(10)
    ax.text(0, 0.08, f'{int(total)}', ha='center', va='center', fontsize=24, fontweight='bold', color='#f8fafc')
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

    grouped = df[['region', 'predicted_revenue', 'lower_bound', 'upper_bound']].copy().fillna(0)
    x = list(range(len(grouped)))
    predicted = grouped['predicted_revenue'].tolist()
    lower = grouped['lower_bound'].tolist()
    upper = grouped['upper_bound'].tolist()

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    _apply_chart_style(ax, f'{rev} Forecasts', f'Predicted {rev.lower()} with confidence range by {reg.lower()}')
    ax.vlines(x, lower, upper, color='#9db4ff', linewidth=8, alpha=0.35, zorder=1)
    ax.scatter(x, predicted, s=120, color='#00d9ff', edgecolor='#1e293b', linewidth=2, zorder=3, label='Predicted')
    ax.plot(x, predicted, color='#667eea', linewidth=2, zorder=2)
    for index, value in enumerate(predicted):
        ax.text(index, value + max(upper) * 0.02, _format_currency_text(value), ha='center', va='bottom', fontsize=9, color='#f8fafc', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['region'], rotation=25, ha='right')
    ax.set_ylabel(rev, color='#94a3b8')
    ax.yaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    ax.legend(frameon=False, loc='upper left')
    
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color('#94a3b8')

    ax.margins(x=0.06)
    fig.tight_layout()
    return fig

def build_revenue_by_product_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0 or 'product' not in df.columns:
        return _empty_chart_figure('No product revenue data available yet.')

    headers = _get_headers_or_default(org_id)
    rev = headers['revenue']
    prod = headers['product']

    grouped = df.groupby('product')['revenue'].sum().sort_values(ascending=False).reset_index()
    colors = ['#764ba2', '#667eea', '#00d9ff', '#4ecdc4', '#f093fb', '#ff6b6b']

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    _apply_chart_style(ax, f'{rev} by {prod}', f'Ranked by total {rev.lower()} contribution')
    bars = ax.barh(grouped['product'], grouped['revenue'], color=colors[:len(grouped)], edgecolor='none', height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel(rev, color='#94a3b8')
    ax.xaxis.set_major_formatter(FuncFormatter(_currency_formatter))
    for bar, value in zip(bars, grouped['revenue'].tolist()):
        ax.text(bar.get_width() + max(grouped['revenue']) * 0.015, bar.get_y() + bar.get_height() / 2, _format_currency_text(value), va='center', ha='left', fontsize=9, color='#e2e8f0', fontweight='bold')
    ax.margins(x=0.02)
    fig.tight_layout()
    return fig

def build_units_by_region_chart(org_id):
    df = _load_sales_frame(org_id)
    if len(df) == 0 or 'region' not in df.columns or 'units_sold' not in df.columns:
        return _empty_chart_figure('No regional units data available yet.')

    headers = _get_headers_or_default(org_id)
    units = headers['units_sold']
    reg = headers['region']

    grouped = df.groupby('region')['units_sold'].sum().sort_values(ascending=False).reset_index()
    colors = ['#4ecdc4', '#ff6b6b', '#667eea', '#00d9ff', '#764ba2', '#f093fb']

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor='#1e293b')
    _apply_chart_style(ax, f'{units} by {reg}', f'Ranked by total {units.lower()} sold')
    bars = ax.bar(grouped['region'], grouped['units_sold'], color=colors[:len(grouped)], edgecolor='none', width=0.52)
    ax.set_ylabel(units, color='#94a3b8')
    for bar, value in zip(bars, grouped['units_sold'].tolist()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(grouped['units_sold']) * 0.015, f"{int(value):,}", va='bottom', ha='center', fontsize=9, color='#e2e8f0', fontweight='bold')
    ax.margins(y=0.08)
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
