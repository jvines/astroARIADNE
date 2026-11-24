"""SEDPlotter settings: float figure sizes and configurable excess-model colours.

A journal column is 3.46 in wide, which the integer-only figsize parser could
not express; and the excess/companion model colours were hard-coded, so a
figure could not follow a house colour scheme.
"""
import types

import astroARIADNE.plotter as P


def _read(tmp_path, text):
    f = tmp_path / 'settings.dat'
    f.write_text(text)
    obj = types.SimpleNamespace(settings_dir=str(f))
    P.SEDPlotter._SEDPlotter__read_config(obj)
    return obj


def test_float_figsize(tmp_path):
    s = _read(tmp_path, 'figsize 3.46,2.9\nhr_figsize 7,5\nfontsize 9\n')
    assert s.figsize == (3.46, 2.9)
    assert s.hr_figsize == (7.0, 5.0)
    assert s.fontsize == 9


def test_excess_colours_default_to_original(tmp_path):
    s = _read(tmp_path, 'fontsize 9\n')
    for k, v in P._EXCESS_COLOR_DEFAULTS.items():
        assert getattr(s, k) == v
    assert s.irx_comp_color == 'tab:red' and s.irx_disk_alt_color == 'seagreen'


def test_excess_colours_override(tmp_path):
    s = _read(tmp_path, 'irx_comp_color #ff00ff\nirx_disk_color #00ffff\n')
    assert s.irx_comp_color == '#ff00ff'
    assert s.irx_disk_color == '#00ffff'
    assert s.irx_disk_alt_color == 'seagreen'


def test_shipped_settings_parse():
    obj = types.SimpleNamespace(settings_dir=None)
    P.SEDPlotter._SEDPlotter__read_config(obj)
    assert obj.figsize == (12.0, 8.0)
    assert obj.irx_comp_color == 'tab:red'


def test_sed_details(tmp_path):
    s = _read(tmp_path, 'legend_fontsize 6\nsed_draws 20\nlegend_loc none\n')
    assert s.legend_fontsize == 6 and s.sed_draws == 20 and s.legend_loc == 'none'
    d = _read(tmp_path, 'fontsize 9\n')
    assert d.legend_fontsize == 9 and d.sed_draws == 100 and d.legend_loc == 'upper right'
