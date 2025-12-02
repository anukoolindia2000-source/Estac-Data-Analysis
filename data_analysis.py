
import pandas as pd
import numpy as np
from plotly.graph_objs import *
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional
from read_bucket import ReadDataFrame
# Data Visualization using plotly
import plotly.express as px
import plotly.io as pio
# pio.renderers.default = "vscode"
pio.renderers.default = "svg"
init_notebook_mode(connected=True)


def plot_corr_heatmap(df, figsize=(10, 8), cmap="coolwarm", annot=True, fmt=".2f", **kwargs):
    """
    Plots a heatmap of the correlation matrix of a dataframe.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe (numeric columns only are recommended)
    figsize : tuple
        Size of the heatmap figure
    cmap : str
        Colormap to use for the heatmap
    annot : bool
        Whether to display correlation values inside each cell
    fmt : str
        Number formatting for annotation text
    **kwargs : dict
        Any additional arguments passed to sns.heatmap()
    """

    # Calculate correlation matrix
    corr = df.corr(numeric_only=True)

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        square=True,
        **kwargs
    )

    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()

def dual_axis_plot(df, x, y1, y2, **kwargs):
    """
    Creates a dual-axis line chart (Plotly).
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    x : str
        Column for x-axis.
    y1 : str
        Primary y-axis column (left).
    y2 : str
        Secondary y-axis column (right).
    **kwargs :
        title, line colors, axis labels, width, height, template, etc.
    """

    # ---- Defaults if not provided ----
    title = kwargs.get("title", f"{y1} and {y2} vs {x}")
    template = kwargs.get("template", "plotly_white")
    width = kwargs.get("width", 1000)
    height = kwargs.get("height", 500)

    y1_color = kwargs.get("y1_color", "red")     # Default primary axis color
    y2_color = kwargs.get("y2_color", "blue")    # Default secondary axis color

    y1_label = kwargs.get("y1_label", y1)
    y2_label = kwargs.get("y2_label", y2)
    x_label  = kwargs.get("x_label", x)

    # ---- Figure with two y-axes ----
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ---- Primary axis trace ----
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y1],
            name=y1,
            line=dict(color=y1_color)
        ),
        secondary_y=False
    )

    # ---- Secondary axis trace ----
    fig.add_trace(
        go.Scatter(
            x=df[x],
            y=df[y2],
            name=y2,
            line=dict(color=y2_color)
        ),
        secondary_y=True
    )

    # ---- Update axes ----
    fig.update_yaxes(
        title_text=f"<b>{y1_label}</b>",
        title_font=dict(color=y1_color),
        secondary_y=False,
        showgrid=False
    )

    fig.update_yaxes(
        title_text=f"<b>{y2_label}</b>",
        title_font=dict(color=y2_color),
        secondary_y=True,
        showgrid=True
    )

    fig.update_xaxes(
        title_text=f"<b>{x_label}</b>",
        showgrid=True
    )

    # ---- Layout ----
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        template=template
    )

    fig.show()



def plot_line_multi(df, x, y, **kwargs):
    """
    Generic Plotly line chart function.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.
    x : str
        Column to use on x-axis.
    y : list
        List of column names to plot on y-axis.
    **kwargs : 
        Additional parameters like:
        - title
        - width
        - height
        - template
        - line_width
        - marker_size
        ...and anything else for flexibility.
    """
    
    # Defaults (if not provided)
    title = kwargs.get("title", "Line Plot")
    template = kwargs.get("template", "plotly_white")
    width = kwargs.get("width", 900)
    height = kwargs.get("height", 500)
    line_width = kwargs.get("line_width", 3)
    marker_size = kwargs.get("marker_size", 8)

    fig = go.Figure()

    for col in y:
        fig.add_trace(go.Scatter(
            x=df[x],
            y=df[col],
            mode='lines+markers',
            name=col,
            line=dict(width=line_width),
            marker=dict(size=marker_size)
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title="Value",
        template=template,
        width=width,
        height=height
    )

    fig.show()



def plot_diverging_plot(
    df, 
    x_col, 
    y1_col, 
    y2_col, 
    **kwargs
):
    """
    Create a dual-axis plot (temperature rate vs humidity rate)
    using Plotly with configurable customization via kwargs.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    x_col : str
        Column for x-axis (time)
    y1_col : str
        Primary y-axis column (Temperature rate)
    y2_col : str
        Secondary y-axis column (Humidity rate)
    **kwargs : dict
        Any extra keyword arguments for customizing:
            - title
            - y1_name
            - y2_name
            - width
            - height
            - line1_color
            - line2_color
            - dash2
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Defaults (overridden by kwargs)
    title = kwargs.get("title", "Temperature & Humidity Rate Over Time")
    y1_name = kwargs.get("y1_name", "Temp Rate (°C/hr)")
    y2_name = kwargs.get("y2_name", "Humidity Rate (%/hr)")
    line1_color = kwargs.get("line1_color", "red")
    line2_color = kwargs.get("line2_color", "blue")
    dash2 = kwargs.get("dash2", "dot")
    width = kwargs.get("width", 1100)
    height = kwargs.get("height", 500)

    # --- Primary Y Axis Trace ---
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y1_col],
            mode="lines",
            name=y1_name,
            line=dict(color=line1_color, width=2),
            hovertemplate="Time: %{x}<br>"+y1_name+": %{y:.2f}<extra></extra>"
        ),
        secondary_y=False
    )

    # --- Secondary Y Axis Trace ---
    fig.add_trace(
        go.Scatter(
            x=df[x_col],
            y=df[y2_col],
            mode="lines",
            name=y2_name,
            line=dict(color=line2_color, width=2, dash=dash2),
            hovertemplate="Time: %{x}<br>"+y2_name+": %{y:.2f}<extra></extra>"
        ),
        secondary_y=True
    )

    # Common Y-range
    y_min = min(df[y1_col].min(), df[y2_col].min())
    y_max = max(df[y1_col].max(), df[y2_col].max())

    # Update axes
    fig.update_yaxes(title_text=y1_name, range=[y_min, y_max], secondary_y=False)
    fig.update_yaxes(title_text=y2_name, range=[y_min, y_max], secondary_y=True)
    fig.update_xaxes(title_text="Time")

    # Horizontal zero lines
    fig.add_hline(y=0, line=dict(color="black", dash="dash"), opacity=0.6, secondary_y=False)
    fig.add_hline(y=0, line=dict(color="grey", dash="dot"), opacity=0.6, secondary_y=True)

    # Layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        template="plotly_white"
    )

    fig.show()


def plot_ma_with_phase(
    df, 
    x_col, 
    y_col, 
    phase_col, 
    **kwargs
):
    """
    Plot a moving average line + colored markers by Heating/Cooling phase.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe
    x_col : str
        X-axis column
    y_col : str
        Temperature moving average column
    phase_col : str
        Column with categorical phases (e.g., Heating / Cooling)
    **kwargs : dict
        Optional custom styling such as:
        - title
        - ma_color
        - marker_heating_color
        - marker_cooling_color
        - marker_size
        - width, height
    """

    # Defaults that can be overridden via kwargs
    title = kwargs.get("title", "Temperature MA Colored by Phase")
    ma_color = kwargs.get("ma_color", "green")
    marker_heating_color = kwargs.get("marker_heating_color", "red")
    marker_cooling_color = kwargs.get("marker_cooling_color", "blue")
    marker_size = kwargs.get("marker_size", 6)
    width = kwargs.get("width", 1100)
    height = kwargs.get("height", 500)

    # Map phase colors dynamically
    phase_color_map = {
        'Heating': marker_heating_color,
        'Cooling': marker_cooling_color
    }

    # --- Create figure ---
    fig = go.Figure()

    # 1. MA Line
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines',
        name=f"{y_col} (MA)",
        line=dict(color=ma_color, width=1.8)
    ))

    # 2. Markers colored by phase
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        name="Phase",
        marker=dict(
            size=marker_size,
            color=df[phase_col].map(phase_color_map),
            line=dict(width=1, color="DarkSlateGrey")
        ),
        customdata=df[[phase_col]].values,
        hovertemplate=(
            "Time: %{x}<br>"
            f"{y_col}: "+"%{y:.2f}<br>"
            f"Phase: "+"%{customdata[0]}<extra></extra>"
        )
    ))

    # Layout
    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        width=width,
        height=height,
        legend=dict(yanchor="top", y=1.15, xanchor="right", x=0.99)
    )

    fig.show()


def plot_timeline(data: pd.DataFrame,start_col: str,end_col: str,phase_col: str,color_map: dict,show: bool = False,**kwargs):
    """
    Create a timeline chart showing temperature phases (e.g., Heating/Cooling).

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing start, end, and phase columns.
    start_col : str, default='start_time'
        Column representing the start of each phase.
    end_col : str, default='end_time'
        Column representing the end of each phase.
    phase_col : str, default='phase'
        Column representing the phase category (e.g., Heating, Cooling).
    color_map : dict, optional
        Custom color mapping for phases, e.g., {'Heating':'red', 'Cooling':'blue'}.
    **kwargs :
        Additional keyword arguments forwarded to px.timeline().
        Examples: title, hover_data, height, template, category_orders, etc.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly timeline figure.
    """
    try:
        # Default color map if not provided
        if color_map is None:
            color_map = {"Heating": "red", "Cooling": "blue"}

        # Default hover fields
        hover_defaults = {"delta": True, start_col: True, end_col: True}
        hover_data = kwargs.pop("hover_data", hover_defaults)

        fig = px.timeline(
            data,
            x_start=start_col,
            x_end=end_col,
            y=phase_col,
            color=phase_col,
            color_discrete_map=color_map,
            hover_data=hover_data,
            **kwargs
        )

        # Order y-axis (Heating above Cooling)
        fig.update_yaxes(categoryorder="array", categoryarray=["Heating", "Cooling"])

        # Layout customization
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Phase Type",
            template=kwargs.pop("template", "plotly_white"),
            height=kwargs.pop("height", 400)
        )

        if show:
            fig.show()
        return fig
    except Exception as e:
        print(f"Error creating temperature timeline: {e}")
        raise e
    
def plot_relationship(data: pd.DataFrame, col1: str, col2: str, color: Optional[str]=None,hover_data: Optional[list]=None):
    """
    Return an interactive scatter plotly figure.
    trendline can be 'ols' to show linear regression (uses statsmodels if installed) or None.
    """
    for col in (col1, col2):
        if col not in data.columns:
            raise KeyError(f"Column '{col}' not found.")
    fig = px.scatter(data, x=col1, y=col2, color=color, hover_data=hover_data,
                     labels={col1: col1, col2: col2}, title=f"{col1.capitalize()} vs {col2.capitalize()}")
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(template='plotly_white')
    return fig


def get_distribution_type(data: pd.DataFrame, feature: str, nbins: int = 30, marginal: Optional[str] = None):
    """
    Return a Plotly Figure (histogram) for distribution.
    marginal: 'rug' | 'box' | None
    """
    if feature not in data.columns:
        raise KeyError(f"Feature '{feature}' not found.")
    fig = px.histogram(data, x=feature, nbins=nbins, marginal=marginal,
                       title=f"Distribution of {feature}", labels={feature: feature})
    fig.update_layout(bargap=0.05, template='plotly_white')
    return fig


def check_outliers(data: pd.DataFrame, feature: str, **kwargs):
    """
    Visualize outliers for a given feature using a Seaborn boxplot.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the feature.
    feature : str
        The column name to visualize for outliers.
    **kwargs :
        Additional keyword arguments to customize seaborn.boxplot().
        Examples: color='skyblue', orient='v', width=0.5, showcaps=False
    """
    try:
        plt.figure(figsize=kwargs.pop('figsize', (6, 3)))  # optional figure size
        sns.boxplot(data=data, x=feature, **kwargs)
        plt.title(f"Outlier Check: {feature}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting feature '{feature}': {e}")
        raise