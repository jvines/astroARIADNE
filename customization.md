# Plot customization
**ARIADNE** offers publication ready plots, but if you don't like how they look, you can tweak some stuff!

In order to customize your plots, first you need to create settings file or just copy the default one and modify it. The default settings can be found in [this file](https://github.com/jvines/astroARIADNE/blob/master/astroARIADNE/Datafiles/plot_settings.dat).

## General settings
These are some general settings that apply to (almost) every plot, they are the figure size, the font size, the font name, and the tick label size

In our file we just add the lines
```
figsize w,h
fontsize size
fontname name
tick_labelsize size
```

## SED plot settings
These settings apply to the SED plot with the model

```
scatter_size size
marker marker
marker_model marker
marker_colors color
marker_colors_model color
scatter_alpha alpha
model_color color
error_color color
error_alpha alpha
```

- `scatter_size` controls the marker sizes of both the synthetic and original photometry in the plot
- `marker` corresponds to the marker for the original photometry. You can find a list of markers [here](https://matplotlib.org/api/markers_api.html)
- `marker_model` corresponds to the marker for the synthetic photometry.
- `marker_colors` is the color for the original photometry. You can find the possible colors [here](https://matplotlib.org/gallery/color/named_colors.html)
- `marker_colors_model` is the color for the synthetic photometry.
- `scatter_alpha` is the alpha or transparency value for the original photometry markers. Value must be between 0 and 1 where 1 is opaque and 0 transparent.
- `model_color` is the color for the model line.
- `error_color` is the color for the original photometry errorbars
- `error_alpha` is the alpha or transparency value for the original photometry errorbars.

## Corner plot settings
These settings apply to the corner plot

```
corner_med_c color
corner_med_style style
corner_v_c lightcoral
corner_v_style style
corner_fontsize size
corner_tick_fontsize size
corner_labelpad pad
corner_marker marker
```

- `corner_med_c` is the color for the median value.
- `corner_med_style` is the linestyle for the median value. You can find a list of styles [here](https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html)
- `corner_v_c` is the color for the 1$\sigma$ values.
- `corner_v_style` is the linestyle for the 1$\sigma$ values.
- `corner_fontsize` is the fontsize for the titles and axis labels of the corner plot
- `corner_tick_fontsize` is the corner plot tick's font size
- `corner_labelpad` is the padding value for the axis labels, in case they are too close to the tick labels
- `corner_marker` is the marker for the median value.

## HR Diagram settings
These settings apply to the HR diagram plot.

```
hr_figsize w,h
hr_marker marker
hr_color color
hr_cmap cmap
```

- `hr_figsize` is the figure size in width,height
- `hr_marker` is the marker of the star in the diagram
- `hr_color` is the marker color in the diagram
- `hr_cmap` is the color map to show mass values. You can find a list of colormaps [here](https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)

Now that we have our settings file ready, we need to give it to the `SEDPlotter` constructor:

```python
from astroARIADNE.plotter import SEDPlotter

in_file = 'output/folder/BMA_out.pkl'
plots_out_folder = 'your plots folder here'

artist = SEDPlotter(in_file, plots_out_folder, settings='dir to your settings file')
```

And voilà! Our plots will now look exactly as we want them to!