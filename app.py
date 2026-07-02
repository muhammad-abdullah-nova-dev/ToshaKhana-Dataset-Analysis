import flask
from flask import Flask, render_template
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
from matplotlib import gridspec
import io
import base64
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import numpy as np
import scipy.stats as stats

app = Flask(__name__)
# In-memory cache for plots
plot_cache = {}
# Load and preprocess dataset (exact notebook code)
def load_data():
    try:
        df = pd.read_csv('static\\Refined_TK_dataver2.csv')
        df['Affiliation'] = df['Affiliation'].fillna('Unknown')
        df['Affiliation'] = df['Affiliation'].replace({
            'Gen Mus': 'Gen. Musharraf',
            'Gen Mush': 'Gen. Musharraf',
            'Gen. Musharrafarraf': 'Gen. Musharraf'
        })
        df['Detail of Gifts'] = df['Detail of Gifts'].replace({'One Carpet': 'One carpet'})
        pd.options.display.float_format = '{:,.0f}'.format
        df['Assessed Value'] = pd.to_numeric(df['Assessed Value'], errors='coerce')
        df['Retention Cost'] = pd.to_numeric(df['Retention Cost'], errors='coerce')
        df.dropna(subset=['Assessed Value', 'Retention Cost'], how='all', inplace=True)
        return df, None
    except FileNotFoundError:
        return None, "Error: 'Refined_TK_dataver2.csv' not found in the project directory."
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

# Helper function to convert plot to base64
def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)  # Reduced DPI for performance
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close('all')  # Close all figures to free memory
    return img_str
# Compute probability distributions
def compute_probabilities(df):
    # Probability of retention
    retained_prob = df['Retained'].value_counts(normalize=True).get('Yes', 0)
    
    # Set pandas display format
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Conditional probability of non-retention by Affiliation
    affiliation_group = df.groupby('Affiliation')['Retained'].value_counts(normalize=True).unstack()
    affiliation_probs = affiliation_group['No'].dropna().sort_values(ascending=False).to_dict()
    
    # Posterior probabilities of non-retention by Item Category
    category_group = df.groupby('Item Category')['Retained'].value_counts(normalize=True).unstack()
    category_probs = category_group['No'].dropna().sort_values(ascending=False).to_dict()
    
    return retained_prob, affiliation_probs, category_probs
# Generate all seven graphs
def generate_graphs(df):
    cache_key = 'graphs'
    if cache_key in plot_cache:
        return plot_cache[cache_key]
    
    plots = []
    
    try:
        # Graph 1 & 2: Main and Inset Histograms
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax_main = plt.subplot(gs[0])
        ax_inset = plt.subplot(gs[1])
        ax_main.hist(df['Assessed Value'].dropna(), bins=100, color='skyblue', edgecolor='black')
        ax_main.set_title("Full Distribution (Compressed)")
        ax_main.set_xlabel("Assessed Value (PKR)")
        ax_main.set_ylabel("Frequency")
        assessed_99 = df['Assessed Value'].quantile(0.90)
        filtered = df[df['Assessed Value'] <= assessed_99]
        ax_inset.hist(filtered['Assessed Value'], bins=50, color='orange', edgecolor='black')
        ax_inset.set_title("Zoomed-in (0–90%)")
        ax_inset.set_xlabel("Assessed Value")
        plt.tight_layout()
        plots.append({'title': 'Assessed Value Histograms', 'image': plot_to_base64()})

        # Graph 3: Item Category Bar Plot
        plt.figure(figsize=(10, 6))
        df['Item Category'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
        plt.title('Item Category Distribution')
        plt.xlabel('Item Category')
        plt.ylabel('Count')
        plots.append({'title': 'Item Category Distribution', 'image': plot_to_base64()})

        # Graph 4: Affiliation Bar Plot
        plt.figure(figsize=(10, 6))
        df['Affiliation'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
        plt.title('Affiliation Distribution')
        plt.xlabel('Affiliation')
        plt.ylabel('Count')
        plots.append({'title': 'Affiliation Distribution', 'image': plot_to_base64()})

        # Graph 5: Assessed Value Over Time
        df_time = df.copy()
        df_time['Date'] = pd.to_datetime(df_time['Date'], errors='coerce')
        df_time = df_time.dropna(subset=['Date', 'Assessed Value'])
        df_time = df_time.sort_values('Date')
        plt.figure(figsize=(10, 6))
        plt.plot(df_time['Date'], df_time['Assessed Value'], marker='o', linestyle='-', color='b')
        plt.ticklabel_format(style='plain', axis='y')
        plt.title('Assessed Value of Gifts Over Time')
        plt.xlabel('Date')
        plt.ylabel('Assessed Value')
        plt.grid(True)
        plots.append({'title': 'Assessed Value of Gifts Over Time', 'image': plot_to_base64()})

        # Graph 6: Pie Chart of Selected Affiliations
        filtered_df = df[df['Affiliation'].isin(['PTI', 'PMLN', 'PPP', 'Bureaucracy', 'Media', 'Police', 'Military', 'Gen. Musharraf'])]
        affiliation_counts = filtered_df['Affiliation'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(affiliation_counts, labels=affiliation_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'])
        plt.title('Number of Gifts Received')
        plots.append({'title': 'Number of Gifts Received', 'image': plot_to_base64()})

        # Graph 7: Pie Chart of PTI and PMLN
        filtered_df = df[df['Affiliation'].isin(['PTI', 'PMLN'])]
        affiliation_counts = filtered_df['Affiliation'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(affiliation_counts, labels=affiliation_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff'])
        plt.title('Number of Gifts Received During PTI and PMLN Eras')
        plots.append({'title': 'Number of Gifts Received During PTI and PMLN Eras', 'image': plot_to_base64()})

        # Graph 8: Correlation Heatmap
        df_corr = df.copy()
        df_corr['Retained_Binary'] = df_corr['Retained'].apply(
            lambda x: 1 if str(x).strip().lower() in ('yes', 'retained') else 0
        )
        df_corr['Date_Parsed'] = pd.to_datetime(df_corr['Date'], errors='coerce')
        df_corr['Year'] = df_corr['Date_Parsed'].dt.year

        numeric_cols = ['Assessed Value', 'Retention Cost', 'Retained_Binary', 'Year']
        display_labels = ['Assessed Value', 'Retention Cost', 'Retained', 'Year']
        corr_data = df_corr[numeric_cols].dropna()
        corr_matrix = corr_data.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(display_labels, fontsize=11)

        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                val = corr_matrix.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.4f}',
                        ha='center', va='center', color=color, fontsize=13, fontweight='bold')

        plt.colorbar(im, label='Correlation Coefficient')
        plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plots.append({'title': 'Correlation Heatmap', 'image': plot_to_base64()})

        # Graph 9: Covariance Heatmap
        df_cov = df.copy()
        df_cov['Retained_Binary'] = df_cov['Retained'].apply(
            lambda x: 1 if str(x).strip().lower() in ('yes', 'retained') else 0
        )
        df_cov['Date_Parsed'] = pd.to_datetime(df_cov['Date'], errors='coerce')
        df_cov['Year'] = df_cov['Date_Parsed'].dt.year

        cov_cols = ['Assessed Value', 'Retention Cost', 'Retained_Binary', 'Year']
        cov_labels = ['Assessed Value', 'Retention Cost', 'Retained', 'Year']
        cov_data = df_cov[cov_cols].dropna()
        cov_matrix = cov_data.cov()

        fig, ax = plt.subplots(figsize=(10, 8))
        log_abs = np.log10(np.abs(cov_matrix.values) + 1)
        max_val = log_abs.max()
        im = ax.imshow(log_abs, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max_val)

        ax.set_xticks(range(len(cov_cols)))
        ax.set_yticks(range(len(cov_cols)))
        ax.set_xticklabels(cov_labels, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(cov_labels, fontsize=11)

        for i in range(len(cov_matrix)):
            for j in range(len(cov_matrix)):
                val = cov_matrix.iloc[i, j]
                if abs(val) >= 1e6:
                    txt = f'{val:.2e}'
                elif abs(val) >= 1:
                    txt = f'{val:,.2f}'
                else:
                    txt = f'{val:.4f}'
                brightness = log_abs[i, j] / max_val if max_val > 0 else 0
                color = 'white' if brightness > 0.6 else 'black'
                ax.text(j, i, txt,
                        ha='center', va='center', color=color, fontsize=10, fontweight='bold')

        plt.colorbar(im, label='log\u2081\u2080(|Covariance| + 1)')
        plt.title('Covariance Matrix of Numerical Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plots.append({'title': 'Covariance Matrix', 'image': plot_to_base64()})

        plot_cache[cache_key] = plots
        return plots
    except Exception as e:
        return [{'title': 'Error', 'image': None, 'error': f"Error generating graphs: {str(e)}"}]
# Compute linear regression
def compute_regression(df):
    cache_key = 'regression'
    if cache_key in plot_cache:
        return plot_cache[cache_key]
    
    try:
        # Drop rows with missing values
        df_reg = df.dropna(subset=['Assessed Value', 'Retention Cost'])
        
        # Check if enough data remains
        if len(df_reg) < 2:
            return None, None, "Error: Insufficient data for regression after dropping missing values."
        
        # Prepare data
        X = df_reg[['Assessed Value']]  # 2D array
        y = df_reg['Retention Cost']    # 1D array
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict
        y_pred = model.predict(X)
        
        # Generate plot
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
        plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
        plt.title("Linear Regression: Retention Cost vs Assessed Value")
        plt.xlabel("Assessed Value (PKR)")
        plt.ylabel("Retention Cost (PKR)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Convert plot to base64
        img_str = plot_to_base64()
        
        # Model statistics
        stats = {
            'intercept': model.intercept_,
            'slope': model.coef_[0],
            'r2': r2_score(y, y_pred)
        }
        
        result = (img_str, stats, None)
        plot_cache[cache_key] = result
        return result
    except Exception as e:
        return None, None, f"Error in regression modeling: {str(e)}"

# ── ML helpers ────────────────────────────────────────────────────────────────

def train_models(df):
    """Train 3 ML models on the cleaned dataframe. Returns (models_dict, encoders_dict)."""
    try:
        dft = df.copy()

        # Fill nulls for categorical columns
        dft['Item Category'] = dft['Item Category'].fillna('Unknown')
        dft['Affiliation']   = dft['Affiliation'].fillna('Unknown')

        # Label-encode categorical columns
        le_affiliation = LabelEncoder()
        le_category    = LabelEncoder()
        dft['Aff_enc'] = le_affiliation.fit_transform(dft['Affiliation'].astype(str))
        dft['Cat_enc'] = le_category.fit_transform(dft['Item Category'].astype(str))

        # ── Model 1: Linear Regression — Retention Cost ──────────────────────
        df1 = dft.dropna(subset=['Assessed Value', 'Retention Cost'])
        X1  = df1[['Assessed Value']]
        y1  = df1['Retention Cost']
        model_lr = LinearRegression()
        model_lr.fit(X1, y1)

        # ── Model 2: Logistic Regression — Retention Probability ─────────────
        dft['retained_bin'] = (
            dft['Retained'].astype(str).str.strip().str.lower()
            .str.contains('yes|retained', na=False)
        ).astype(int)
        df2 = dft.dropna(subset=['Assessed Value', 'Aff_enc', 'Cat_enc'])
        X2  = df2[['Assessed Value', 'Aff_enc', 'Cat_enc']]
        y2  = df2['retained_bin']
        model_logr = LogisticRegression(max_iter=1000, random_state=42)
        model_logr.fit(X2, y2)

        # ── Model 3: Random Forest — Value Tier ──────────────────────────────
        dft['value_tier'] = pd.cut(
            dft['Assessed Value'],
            bins=[-1, 4999, 39999, 499999, float('inf')],
            labels=['Budget', 'Standard', 'Premium', 'Luxury']
        )
        df3 = dft.dropna(subset=['value_tier', 'Aff_enc', 'Cat_enc'])
        X3  = df3[['Aff_enc', 'Cat_enc']]
        y3  = df3['value_tier'].astype(str)
        model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        model_rf.fit(X3, y3)

        models   = {'lr': model_lr, 'logr': model_logr, 'rf': model_rf}
        encoders = {'affiliation': le_affiliation, 'category': le_category}
        return models, encoders

    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")


def run_predictions(df, models, encoders):
    """Apply all 3 models in batch (vectorized). Returns list of result dicts."""
    dft = df.copy().reset_index(drop=True)
    dft['Item Category'] = dft['Item Category'].fillna('Unknown')
    dft['Affiliation']   = dft['Affiliation'].fillna('Unknown')

    le_aff = encoders['affiliation']
    le_cat = encoders['category']

    # Safe vectorized transform — unseen labels map to 0
    known_aff = set(le_aff.classes_)
    known_cat = set(le_cat.classes_)
    dft['Aff_enc'] = [le_aff.transform([v])[0] if v in known_aff else 0
                      for v in dft['Affiliation'].astype(str)]
    dft['Cat_enc'] = [le_cat.transform([v])[0] if v in known_cat else 0
                      for v in dft['Item Category'].astype(str)]

    # ── Model 1: batch predict retention cost ────────────────────────────────
    mask1 = dft['Assessed Value'].notna()
    pred_rc_arr = np.zeros(len(dft), dtype=float)
    if mask1.any():
        pred_rc_arr[mask1] = models['lr'].predict(dft.loc[mask1, ['Assessed Value']])
    dft['pred_rc'] = pred_rc_arr.astype(int)

    # ── Model 2: batch predict retention probability ──────────────────────────
    mask2 = dft['Assessed Value'].notna()
    prob_arr = np.full(len(dft), 0.5)
    if mask2.any():
        X2 = dft.loc[mask2, ['Assessed Value', 'Aff_enc', 'Cat_enc']]
        prob_arr[mask2] = models['logr'].predict_proba(X2)[:, 1]
    dft['ret_prob'] = np.round(prob_arr, 4)

    # ── Model 3: batch predict value tier ────────────────────────────────────
    X3 = dft[['Aff_enc', 'Cat_enc']]
    try:
        dft['tier'] = models['rf'].predict(X3)
    except Exception:
        dft['tier'] = 'Standard'

    # ── Build result list ─────────────────────────────────────────────────────
    rows = []
    for i, row in dft.iterrows():
        av = row['Assessed Value']
        rc = row.get('Retention Cost', None)
        rows.append({
            'row_index':             i,
            'detail':                str(row.get('Detail of Gifts', ''))[:40],
            'recipient':             str(row.get('Name of Recipient', ''))[:30],
            'affiliation':           str(row.get('Affiliation', 'Unknown')),
            'category':              str(row.get('Item Category', 'Unknown')),
            'assessed_value':        int(av) if pd.notna(av) else 0,
            'actual_retention_cost': int(rc) if pd.notna(rc) else 0,
            'pred_retention_cost':   int(row['pred_rc']),
            'retention_probability': float(row['ret_prob']),
            'value_tier':            str(row['tier']),
            'actual_retained':       str(row.get('Retained', '')),
        })
    return rows


@app.route('/predict')
def predict():
    df, error = load_data()
    if error:
        return render_template('predict.html', error=error)

    try:
        if 'predict_data' not in plot_cache:
            models, encoders = train_models(df)
            result_rows = run_predictions(df, models, encoders)
            plot_cache['predict_data'] = result_rows

        result_rows = plot_cache['predict_data']

        total = len(result_rows)
        summary = {
            'total_rows':          total,
            'avg_retention_prob':  round(
                sum(r['retention_probability'] for r in result_rows) / total * 100, 1
            ) if total else 0,
            'tier_counts': {
                'Budget':   sum(1 for r in result_rows if r['value_tier'] == 'Budget'),
                'Standard': sum(1 for r in result_rows if r['value_tier'] == 'Standard'),
                'Premium':  sum(1 for r in result_rows if r['value_tier'] == 'Premium'),
                'Luxury':   sum(1 for r in result_rows if r['value_tier'] == 'Luxury'),
            },
            'high_risk_count': sum(1 for r in result_rows if r['retention_probability'] < 0.3),
        }

        return render_template('predict.html', rows=result_rows, summary=summary)

    except Exception as e:
        return render_template('predict.html', error=f"Prediction error: {str(e)}")


@app.route('/predict_ci', methods=['GET', 'POST'])
def predict_ci():
    df, error = load_data()
    if error:
        return render_template('predict_ci.html', error=error)
    
    prediction = None
    if flask.request.method == 'POST':
        try:
            val_str = flask.request.form.get('assessed_value', '').strip()
            if not val_str:
                return render_template('predict_ci.html', error="Please enter an Assessed Value.")
                
            assessed_val = float(val_str)
            
            df_reg = df.dropna(subset=['Assessed Value', 'Retention Cost'])
            if len(df_reg) < 2:
                return render_template('predict_ci.html', error="Not enough data points.")
                
            X = df_reg[['Assessed Value']].values
            y = df_reg['Retention Cost'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Point Predict
            x0 = np.array([[assessed_val]])
            y_pred = model.predict(x0)[0]
            
            # Prediction Interval
            n = len(X)
            x_mean = np.mean(X)
            ss_x = np.sum((X - x_mean)**2)
            
            y_fit = model.predict(X)
            sse = np.sum((y - y_fit)**2)
            s_e = np.sqrt(sse / (n - 2))
            
            se_pred = s_e * np.sqrt(1 + 1/n + (assessed_val - x_mean)**2 / ss_x)
            t_crit = stats.t.ppf(0.975, df=n-2)
            
            margin_error = t_crit * se_pred
            
            lower_bound = max(0, y_pred - margin_error)
            upper_bound = y_pred + margin_error
            
            prediction = {
                'assessed_value': assessed_val,
                'predicted_cost': y_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'margin_error': margin_error,
                'confidence': 95
            }
            
        except ValueError:
            error = "Invalid input. Please enter a numerical value."
        except Exception as e:
            error = f"Error processing prediction: {str(e)}"
            
    return render_template('predict_ci.html', prediction=prediction, error=error)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/stats')
def stats():
    df, error = load_data()
    if error:
        return render_template('stats.html', error=error)
    stats_df = df[['Assessed Value', 'Retention Cost']].describe()
    return render_template('stats.html', stats=stats_df.to_dict())

@app.route('/graphs')
def graphs():
    df, error = load_data()
    if error:
        return render_template('graphs.html', error=error)
    
    plots = generate_graphs(df)
    if plots and 'error' in plots[0]:
        return render_template('graphs.html', error=plots[0]['error'])
    
    return render_template('graphs.html', plots=plots)


# Per-graph detail metadata
GRAPH_DETAILS = [
    {
        'slug': 'assessed-value-histograms',
        'title': 'Assessed Value Histograms',
        'subtitle': 'Distribution of gift assessed values across the full dataset',
        'chart_type': 'Histogram',
        'accent': '#C9A84C',
        'description': (
            'This dual-panel histogram visualises how assessed gift values are distributed across '
            'the entire Toshakhana dataset. The left panel shows the full compressed distribution — '
            'the extreme right skew immediately reveals that a small number of very high-value gifts '
            'dominate the range, pulling the mean far above the median. The right panel zooms into '
            'the 0–90th percentile range, stripping out those outliers to expose the shape of the '
            'typical gift: most items cluster in the lower value bands, with frequency dropping off '
            'sharply as value increases.'
        ),
        'insights': [
            'The distribution is heavily right-skewed — a handful of luxury items (watches, vehicles, jewellery) account for a disproportionate share of total assessed value.',
            'The median assessed value is significantly lower than the mean, confirming the outlier effect.',
            'The 0–90th percentile view shows the majority of gifts fall in a relatively narrow lower-value band.',
            'High-value outliers are likely linked to state visits and diplomatic exchanges at the highest levels of government.',
        ],
        'x_axis': 'Assessed Value (PKR)',
        'y_axis': 'Frequency (number of gifts)',
    },
    {
        'slug': 'item-category-distribution',
        'title': 'Item Category Distribution',
        'subtitle': 'Frequency of each gift category in the registry',
        'chart_type': 'Bar Chart',
        'accent': '#3498DB',
        'description': (
            'This bar chart breaks down the Toshakhana registry by item category, showing how many '
            'gifts fall into each type. It answers the question: what kinds of gifts are most commonly '
            'received by Pakistani officials? Categories like watches, shawls, and decorative items '
            'tend to dominate, reflecting the nature of diplomatic gifting culture in the region. '
            'The chart makes it easy to spot which categories are rare versus routine.'
        ),
        'insights': [
            'Watches and jewellery consistently appear among the most frequent high-value categories.',
            'Decorative and ceremonial items (e.g., shields, plaques) are common but typically lower in assessed value.',
            'Some categories appear only once or twice, suggesting one-off diplomatic gifts.',
            'The distribution of categories can indicate which types of gifts are most likely to be retained versus auctioned.',
        ],
        'x_axis': 'Item Category',
        'y_axis': 'Count (number of gifts)',
    },
    {
        'slug': 'affiliation-distribution',
        'title': 'Affiliation Distribution',
        'subtitle': 'Number of gifts received by each political or institutional affiliation',
        'chart_type': 'Bar Chart',
        'accent': '#C0392B',
        'description': (
            'This bar chart shows how many Toshakhana gifts are associated with each affiliation — '
            'political parties, military figures, bureaucrats, and other groups. It reveals which '
            'affiliations have the highest representation in the registry, which is closely tied to '
            'which groups held power during the periods covered by the dataset. PTI and PMLN '
            'affiliations tend to dominate given the time span of the data.'
        ),
        'insights': [
            'PTI-affiliated recipients account for a large share of the dataset, reflecting the party\'s tenure in government.',
            'Military and bureaucratic affiliations are consistently present across all time periods.',
            'Smaller affiliations may represent provincial governments or independent officials.',
            'Affiliation is a strong predictor of retention behaviour — some groups retain gifts at higher rates than others.',
        ],
        'x_axis': 'Affiliation',
        'y_axis': 'Count (number of gifts)',
    },
    {
        'slug': 'assessed-value-over-time',
        'title': 'Assessed Value of Gifts Over Time',
        'subtitle': 'Chronological trend of gift values across the dataset period',
        'chart_type': 'Line Chart',
        'accent': '#1ABC9C',
        'description': (
            'This time-series line chart plots each gift\'s assessed value against the date it was '
            'received, revealing temporal patterns in gifting. Spikes in the chart correspond to '
            'periods of high diplomatic activity or specific high-value gifts. The overall trend '
            'shows whether gift values have increased, decreased, or remained stable over time, '
            'and can be cross-referenced with major political events in Pakistan\'s history.'
        ),
        'insights': [
            'Visible spikes correspond to specific high-value diplomatic gifts, often from Gulf states or China.',
            'Gift frequency and value tend to increase during election years and major state visits.',
            'The time axis reveals gaps in record-keeping for certain periods.',
            'Long-term trend analysis suggests gift values have generally increased in nominal PKR terms.',
        ],
        'x_axis': 'Date',
        'y_axis': 'Assessed Value (PKR)',
    },
    {
        'slug': 'gifts-by-affiliation-pie',
        'title': 'Number of Gifts Received',
        'subtitle': 'Proportional share of gifts across key affiliations',
        'chart_type': 'Pie Chart',
        'accent': '#C9A84C',
        'description': (
            'This pie chart shows the proportional share of Toshakhana gifts among the major '
            'affiliations: PTI, PMLN, PPP, Bureaucracy, Media, Police, Military, and Gen. Musharraf. '
            'It gives an at-a-glance view of which groups dominate the registry. The chart filters '
            'to only the most significant affiliations to avoid clutter from minor entries.'
        ),
        'insights': [
            'PTI and PMLN together account for the majority of gifts in the filtered view.',
            'Military and bureaucratic affiliations hold a steady share across all periods.',
            'PPP\'s share reflects its periods in government, particularly at the federal level.',
            'The "Other" category (not shown here) contains many one-off affiliations that collectively add up.',
        ],
        'x_axis': 'N/A (proportional)',
        'y_axis': 'Percentage share of gifts',
    },
    {
        'slug': 'pti-vs-pmln-pie',
        'title': 'Gifts During PTI and PMLN Eras',
        'subtitle': 'Head-to-head comparison of gift counts between the two major parties',
        'chart_type': 'Pie Chart',
        'accent': '#3498DB',
        'description': (
            'This focused pie chart isolates just PTI and PMLN affiliations to directly compare '
            'how many Toshakhana gifts were received during each party\'s tenure. It strips away '
            'all other affiliations to make the comparison as clear as possible. The result shows '
            'which party\'s era saw more gifts enter the registry — a politically significant '
            'finding given the public controversy around Toshakhana disclosures in Pakistan.'
        ),
        'insights': [
            'The split between PTI and PMLN reflects the relative lengths of their respective government tenures in the dataset.',
            'A higher gift count does not necessarily mean higher total value — PTI-era gifts may differ in average value from PMLN-era gifts.',
            'This comparison became politically significant during the 2022–2023 Toshakhana controversy.',
            'Cross-referencing with the time-series chart can reveal which specific periods drove the counts.',
        ],
        'x_axis': 'N/A (proportional)',
        'y_axis': 'Percentage share of gifts',
    },
    {
        'slug': 'correlation-heatmap',
        'title': 'Correlation Heatmap',
        'subtitle': 'Pearson correlation between key numerical features across 4,202 Toshakhana gift records (2002\u20132022)',
        'chart_type': 'Heatmap',
        'accent': '#E74C3C',
        'description': (
            'This heatmap visualises the Pearson correlation coefficients computed from 4,202 complete '
            'Toshakhana gift records spanning 2002 to 2022. Four numerical features are analysed: '
            'Assessed Value (the official valuation of each gift in PKR), Retention Cost (the fee paid '
            'to retain a gift), Retention Status (encoded as binary: 1 if the gift was retained, 0 if '
            'not \u2014 with an overall retention rate of approximately 88.5%), and Year (the year the gift '
            'was received). The strongest relationship in the dataset is between Assessed Value and '
            'Retention Cost (r = 0.7869), confirming that retention fees are largely proportional to '
            'gift value. Notably, the correlation between Year and Retention Status is \u22120.2673, '
            'suggesting that in more recent years, gifts are somewhat less likely to be retained than '
            'in earlier periods. Meanwhile, both Assessed Value and Retention Cost show only very weak '
            'positive correlations with Year (r \u2248 0.08), indicating that gift values have not '
            'significantly increased in real terms across the two decades covered by this dataset.'
        ),
        'insights': [
            'Assessed Value and Retention Cost have the strongest correlation in the dataset at r = 0.7869, confirming that retention fees in the Toshakhana system are broadly proportional to assessed gift value.',
            'Retention Status has a near-zero correlation with Assessed Value (r = \u22120.0360) \u2014 meaning expensive gifts are NOT more likely to be retained than cheaper ones, contrary to common assumptions.',
            'The moderate negative correlation between Year and Retention Status (r = \u22120.2673) reveals a clear trend: gifts received in more recent years (closer to 2022) are less likely to be retained, possibly reflecting increased public scrutiny and tighter regulations over time.',
            'Assessed Value and Year show only a weak positive correlation (r = 0.0855), suggesting that the nominal value of diplomatic gifts has remained relatively stable over the 2002\u20132022 period covered by the dataset.',
        ],
        'x_axis': 'Feature',
        'y_axis': 'Feature',
    },
    {
        'slug': 'covariance-matrix',
        'title': 'Covariance Matrix',
        'subtitle': 'Unstandardised covariance between key numerical features across 4,202 Toshakhana records',
        'chart_type': 'Heatmap',
        'accent': '#F39C12',
        'description': (
            'This heatmap displays the covariance matrix computed from the same 4,202 Toshakhana gift '
            'records used in the correlation analysis. Unlike correlation (which is normalised between '
            '\u22121 and +1), covariance preserves the original units and scales of the variables, making '
            'it useful for understanding the raw magnitude of how two features vary together. '
            'Because covariance values in this dataset span many orders of magnitude \u2014 from fractions '
            'like 0.0999 (Retained variance) to trillions like 1.52\u00d710\u00b9\u00b3 (Assessed Value '
            'variance) \u2014 the colour scale uses a logarithmic transformation (log\u2081\u2080 of absolute '
            'covariance) so that all cells remain visually distinguishable. The dominant cell is the '
            'Assessed Value \u00d7 Retention Cost covariance at 1.90\u00d710\u00b9\u00b2 PKR\u00b2, '
            'reflecting the strong linear relationship between gift valuation and retention fees '
            'in the Toshakhana system.'
        ),
        'insights': [
            'The Assessed Value variance (1.52\u00d710\u00b9\u00b3) is extremely large, confirming the heavy right-skew seen in the histogram \u2014 a few ultra-high-value gifts (watches, vehicles) inflate the spread dramatically.',
            'Covariance between Assessed Value and Retention Cost (1.90\u00d710\u00b9\u00b2) is the largest off-diagonal value, consistent with the strong correlation (r = 0.79) but expressed in original PKR\u00b2 units.',
            'The negative covariance between Year and Retained (\u22120.51) confirms the trend also seen in the correlation matrix: retention rates have declined in more recent years.',
            'Covariance between Assessed Value and Retained is \u221244,306 \u2014 slightly negative, meaning higher-value gifts show a marginal tendency to NOT be retained, though the effect is weak.',
        ],
        'x_axis': 'Feature',
        'y_axis': 'Feature',
    },
]


@app.route('/graphs/<int:graph_id>')
def graph_detail(graph_id):
    if graph_id < 0 or graph_id >= len(GRAPH_DETAILS):
        return render_template('graphs.html', error="Graph not found."), 404

    df, error = load_data()
    if error:
        return render_template('graph_detail.html', error=error)

    plots = generate_graphs(df)
    if plots and 'error' in plots[0]:
        return render_template('graph_detail.html', error=plots[0]['error'])

    plot = plots[graph_id]
    detail = GRAPH_DETAILS[graph_id]
    total = len(GRAPH_DETAILS)

    prev_id = graph_id - 1 if graph_id > 0 else None
    next_id = graph_id + 1 if graph_id < total - 1 else None

    return render_template(
        'graph_detail.html',
        plot=plot,
        detail=detail,
        graph_id=graph_id,
        prev_id=prev_id,
        next_id=next_id,
        total=total,
    )

@app.route('/distributions')
def distributions():
    df, error = load_data()
    if error:
        return render_template('distributions.html', error=error)
    
    retained_prob, affiliation_probs, category_probs = compute_probabilities(df)
    return render_template('distributions.html', 
                         retained_prob=retained_prob, 
                         affiliation_probs=affiliation_probs, 
                         category_probs=category_probs)
@app.route('/regression')
def regression():
    df, error = load_data()
    if error:
        return render_template('regression.html', error=error)
    
    plot, stats, error = compute_regression(df)
    if error:
        return render_template('regression.html', error=error)
    
    return render_template('regression.html', plot=plot, stats=stats)
if __name__ == '__main__':
    app.run(debug=True, threaded=True)