# Allowed filters in ARIADNE
**ARIADNE** uses pyphot (<https://mfouesneau.github.io/pyphot/libcontent.html>) as an interface for photometry, and thus we use the same filter naming convention \<facility\>\<instrument\>\<filter\>

The allowed filters are (in wavelength order):

- Galex\_FUV
- Galex\_NUV
- STROMGREN\_u
- SkyMapper\_u
- SDSS\_u
- GROUND\_JOHNSON\_U
- SkyMapper\_v
- STROMGREN\_v
- TYCHO\_B\_MvB
- GROUND\_JOHNSON\_B
- STROMGREN\_b
- SDSS\_g
- PS1\_g
- SkyMapper\_g
- GaiaDR2v2\_BP
- TYCHO\_V\_MvB
- STROMGREN\_y
- GROUND\_JOHNSON\_V
- SkyMapper\_r
- SDSS\_r
- PS1\_r
- PS1\_w
- KEPLER\_Kp
- GaiaDR2v2\_G
- GROUND\_COUSINS\_R
- NGTS\_I
- SDSS\_i
- PS1\_i
- SkyMapper\_i
- GaiaDR2v2\_RP
- GROUND\_COUSINS\_I
- TESS
- PS1\_z
- SDSS\_z
- SkyMapper\_z
- PS1\_y
- 2MASS\_J
- 2MASS\_H
- 2MASS\_Ks
- WISE\_RSR\_W1
- WISE\_RSR\_W2
- **WISE\_RSR\_W3**
- **WISE\_RSR\_W4**
- **HERSCHEL\_PACS\_BLUE**
- **HERSCHEL\_PACS\_GREEN**
- **HERSCHEL\_PACS\_RED**

**NOTE: WISE W3, W4 and HERSCHEL filters are ONLY available for INFRARED EXCESS VISUALIZATION and NOT for SED FITTING**
## HST filters

The HST WFC3 (UVIS and IR) and ACS/WFC wide, medium, long-pass and
extremely-wide filters can be fitted. Narrow-band and quad filters are not
included. No catalogue query feeds them, so add the photometry yourself through
`mag_dict` or `Star.add_mag`:

- WFC3: `HST_WFC3_F098M`, `F105W`, `F110W`, `F125W`, `F127M`, `F139M`,
  `F140W`, `F153M`, `F160W`, `F200LP`, `F218W`, `F225W`, `F275W`, `F300X`,
  `F336W`, `F350LP`, `F390M`, `F390W`, `F410M`, `F438W`, `F467M`, `F475W`,
  `F475X`, `F547M`, `F555W`, `F600LP`, `F606W`, `F621M`, `F625W`, `F689M`,
  `F763M`, `F775W`, `F814W`, `F845M`, `F850LP` (each as `HST_WFC3_<filter>`)
- ACS/WFC: `HST_ACS_WFC_F435W`, `F475W`, `F550M`, `F555W`, `F606W`, `F625W`,
  `F775W`, `F814W`, `F850LP` (each as `HST_ACS_WFC_<filter>`)

A model grid that does not reach a band leaves it out of that grid's fit (for
example Sonora below 0.4 um).

## Magnitude systems

PanSTARRS, SDSS, GALEX and SkyMapper magnitudes are read as AB, and every other
band as Vega. HST photometry is published as VEGAMAG, ABMAG or STMAG depending
on the source (the Hubble Source Catalog uses ABMAG), so give the system of any
band that is not in its default with `mag_system`:

```python
s = Star('my star', ra, dec, mag_dict={'HST_WFC3_F275W': (14.20, 0.03)},
         mag_system={'HST_WFC3_F275W': 'AB'})
s.add_mag(15.10, 0.02, 'HST_WFC3_F336W', system='ST')
```

The zero points are 3631 Jy for AB, f_lambda = 10^(-0.4 x 21.10)
erg s^-1 cm^-2 A^-1 for ST, and Vega for Vega.
