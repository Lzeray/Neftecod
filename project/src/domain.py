"""Physical roles of the tags -- model_spec.md §2; ТЗ §4; AVT schematics.

The schematics PDF marks the manipulated variables of the chain explicitly:

    avt_furnace_outlet_temp_c, avt_column_top_temp_c, avt_column_pressure_mpa_abs,
    steam_t, ht_feed_flow_m3h, ht_pressure_mpa

and the feed-side disturbances

    crude_density_kgm3, crude_sulfur_wt_pct, crude_water_wt_pct, crude_feed_rate_tph

with the products

    avt_diesel_sulfur_mgkg, avt_diesel_density_kgm3, avt_diesel_t95_c,
    avt_diesel_flow_tph

This module maps those generic names onto the actual tags of this data package
and records, for every choice, the declared КИП description **and** what the
data show.  Two facts force that double bookkeeping:

  * ``crude_density_kgm3`` maps to ``AVT:D10``, which carries 10 non-null rows,
    all of them the value 240.0 -- a DCS bad-value code.  Crude density is
    therefore an *unmeasured* disturbance and lives in the uncertainty set U
    (eq 64), not in the regressor matrix.  Crude sulfur and water content have
    no tag at all, likewise.
  * several ``КИП`` descriptions on the 24-2000 side do not match their data
    (``HDS:T6`` is documented as a sulfur analyser but reads ~363 degC and
    correlates 0.97 with the reactor temperatures; ``HDS:W7``, documented as
    the product sulfur analyser, correlates 0.04 with the ПАК sulfur signal and
    rails at exactly 10.0, the signature of a 0-10 dP transmitter).  Where the
    dictionary and the data disagree the data win and the conflict is written
    to the assumptions register.

model_spec.md §2: a measured temperature is a **state**, not an input; the
manipulated quantity is the *set point*.  Every temperature channel below is
therefore used through the "set point -> actual temperature" FOPTD of eq 32-33
before it reaches the quality model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

AVT_NS, HDS_NS = "AVT", "HDS"


@dataclass(frozen=True)
class Channel:
    key: str                 # namespaced tag, e.g. "HDS:T5"
    role: str                # "u" | "d" | "y" | "aux"
    label: str               # short human name used in the operator report
    unit: str
    schematic_name: str = ""     # name on the AVT schematics, when marked there
    declared: str = ""           # КИП description
    provenance: str = ""         # why this tag was chosen for this role
    setpoint_channel: bool = False   # True: manipulated through a set point


# ---------------------------------------------------------------- inputs u
MANIPULATED: list[Channel] = [
    Channel("HDS:T5", "u", "R-201 outlet temperature (HDS reactor)", "degC",
            declared="Полисеп. Реактор Р-201. Температура ГСС на выходе",
            provenance="Dominant sulfur handle (eq 13-15). Declared and "
                       "observed kind agree (p50 370 degC). Manipulated "
                       "through its set point (model_spec.md §2).",
            setpoint_channel=True),
    Channel("HDS:F15", "u", "Quench flow to R-202", "kg/h",
            declared="Полисеп. Расход квенча в реактор Р-202",
            provenance="Direct handle on the second-bed temperature profile; "
                       "declared and observed kind agree (p50 3400)."),
    Channel("HDS:P3", "u", "HDS reactor-loop pressure", "MPa",
            schematic_name="ht_pressure_mpa",
            declared="Полисеп. Сепаратор С-201. Давление на входе",
            provenance="Marked manipulated on the schematics as "
                       "ht_pressure_mpa; sets p_H2 through eq 17."),
    Channel("HDS:F26", "u", "HDS feed rate (volumetric)", "m3/h",
            schematic_name="ht_feed_flow_m3h",
            declared="Расход сырья на установку (объемный)",
            provenance="Marked manipulated on the schematics as "
                       "ht_feed_flow_m3h; sets LHSV in eq 16."),
    Channel("HDS:F2", "u", "Recycle-gas rate (from C-201)", "nm3/h",
            declared="Газовая схема. Расход на линии от ЦК-201",
            provenance="Gas-to-oil ratio handle. §7 warning: this is a *flow*, "
                       "not a hydrogen partial pressure -- it enters only "
                       "through the identified residual, never as p_H2."),
    Channel("AVT:T55", "u", "AVT furnace P-3 outlet temperature", "degC",
            schematic_name="avt_furnace_outlet_temp_c",
            declared="Температура на выходе из печи П3 (общ. Выход на FIRC2470)",
            provenance="Marked manipulated on the schematics. The tag is a "
                       "controller output (FIRC2470), i.e. a set point.",
            setpoint_channel=True),
    Channel("AVT:T20", "u", "K-2 top temperature", "degC",
            schematic_name="avt_column_top_temp_c",
            declared="Температура верха К2. Выход на FIRC0953",
            provenance="Marked manipulated on the schematics; the tag is the "
                       "output of controller FIRC0953, i.e. a set point.",
            setpoint_channel=True),
    Channel("AVT:P67", "u", "K-2 top pressure", "MPa",
            schematic_name="avt_column_pressure_mpa_abs",
            declared="Давление верха К2.",
            provenance="Marked manipulated on the schematics."),
    Channel("AVT:F29", "u", "Steam to K-2", "t/h",
            schematic_name="steam_t",
            declared="Расход пара в К2",
            provenance="Marked manipulated on the schematics (steam_t); "
                       "stripping steam sets the diesel cut point."),
]

# ---------------------------------------------------------------- disturbances d
MEASURED_DISTURBANCES: list[Channel] = [
    Channel("AVT:F7", "d", "Desalted crude, pass 3", "t/h",
            schematic_name="crude_feed_rate_tph",
            declared="Расход обессоленной нефти (3-й ход)",
            provenance="Crude charge; sums with F8 and F9 to the unit feed."),
    Channel("AVT:F8", "d", "Desalted crude, pass 1", "t/h",
            schematic_name="crude_feed_rate_tph",
            declared="Расход обессоленной нефти (1-й ход)"),
    Channel("AVT:F9", "d", "Desalted crude, pass 2", "t/h",
            schematic_name="crude_feed_rate_tph",
            declared="Расход обессоленной нефти (2-й ход)"),
]

#: Disturbances the package does not measure.  They enter only through the
#: robust uncertainty set U of eq 64 -- never as a fitted regressor.
UNMEASURED_DISTURBANCES: list[dict] = [
    {"name": "crude_density_kgm3", "tag": "AVT:D10",
     "reason": "AVT:D10 has 10 non-null rows, all equal to the bad-value code "
               "240.0; there is no usable crude-density measurement.",
     "status": "[DOP]"},
    {"name": "crude_sulfur_wt_pct", "tag": None,
     "reason": "No tag in the КИП dictionary and no LIMS point for crude "
               "sulfur; the strongest driver of product sulfur is unmeasured.",
     "status": "[DOP]"},
    {"name": "crude_water_wt_pct", "tag": None,
     "reason": "No tag and no LIMS point for crude water content.",
     "status": "[DOP]"},
]

# ---------------------------------------------------------------- outputs y
#: LIMS sampling points as they appear in the workbook (exact strings).
LIMS_POINTS = {
    "avt_diesel_1": "Установка 'АВТ'. Точка отбора '1'. Продукт 'Дизельное топливо'",
    "avt_diesel_2": "Установка 'АВТ'. Точка отбора '2'. Продукт 'Дизельное топливо'",
    "avt_diesel_21": "Установка 'АВТ'. Точка отбора '2.1'. Продукт 'Дизельное топливо'",
    "avt_diesel_3": "Установка 'АВТ'. Точка отбора '3'. Продукт 'Дизельное топливо'",
    "hds_feed": "Установка 'Гидроочистка'. Точка отбора '1'. Продукт 'ФРАКЦ_ДИЗ'.",
    "hds_product": "Установка 'Гидроочистка'.. Точка отбора '2'. Продукт 'Дизельное топливо'",
}

#: Feed sulfur is reported at the HDS feed point under a *different* parameter
#: name and unit than the product (% mass vs mg/kg); it is the single most
#: important measured disturbance and has only ~130 lab results.
FEED_SULFUR_PARAM = "Mass.Sulfur"
FEED_SULFUR_UNIT = "% масс."

#: The controlled product: hydrotreated diesel leaving unit 24-2000.
PRODUCT_POINT = LIMS_POINTS["hds_product"]
FEED_POINT = LIMS_POINTS["hds_feed"]

#: ПАК stream analysers.
PAK_SULFUR = "24-2000:Mg.Sulfur"
PAK_D15 = "24-2000:D15"

#: Quality parameters carried through the whole system.
QUALITY_PARAMS = {
    "sulfur": {"lims_param": "Mg.Sulfur", "unit": "мг/кг",
               "vak": None, "spec": ("max", 10.0), "hard": True},
    "D15": {"lims_param": "D15", "unit": "кг/м3",
            "vak": "24-2000:GODT:D15", "spec": ("range", (820.0, 845.0)),
            "hard": False},
    "T95": {"lims_param": "95%.T", "unit": "°С",
            "vak": "24-2000:GODT:T95", "spec": ("max", 360.0), "hard": False},
    "T50": {"lims_param": "50%.T", "unit": "°С",
            "vak": "24-2000:GODT:T50", "spec": None, "hard": False},
    "T90": {"lims_param": "90%.T", "unit": "°С",
            "vak": "24-2000:GODT:T90", "spec": None, "hard": False},
    "CFPP": {"lims_param": "CFPP", "unit": "°С",
             "vak": "24-2000:GODT:CFPP", "spec": ("max", -5.0), "hard": False},
    "CloudPoint": {"lims_param": "CloudPoint", "unit": "°С",
                   "vak": "24-2000:GODT:CloudPoint", "spec": None, "hard": False},
    "FlashPoint": {"lims_param": "FlashPoint", "unit": "°С",
                   "vak": None, "spec": ("min", 55.0), "hard": False},
    "CetaneNumber": {"lims_param": "CetaneNumber", "unit": "ед.цет.ч.",
                     "vak": None, "spec": ("min", 51.0), "hard": False},
    "IBP": {"lims_param": "IBP.T", "unit": "°С",
            "vak": "24-2000:GODT:IBP", "spec": None, "hard": False},
    "I250": {"lims_param": "I250", "unit": "% об.",
             "vak": "24-2000:GODT:I250", "spec": None, "hard": False},
}

#: ``LIMS:...`` identifiers used inside the ВАК formulas.  Which sampling point
#: "Pipeline" refers to is not stated anywhere in the package, so both readings
#: are scored against held-out LIMS data in src/validate/vak_audit.py and the
#: better one is used; the choice is recorded in the assumptions register.
#: The product-point option deliberately binds to the *previous* lab result
#: (``LIMSPREV:*``): binding to the current one would let the formula read its
#: own target and would score as skill that does not exist in operation.
VAK_LIMS_CANDIDATES = {
    "LIMS_24_2000_Pipeline_D15": [("hds_feed", "D15"), ("hds_product_prev", "D15")],
    "LIMS_24_2000_Pipeline_95_T": [("hds_feed", "95%.T"),
                                   ("hds_product_prev", "95%.T")],
}

# ---------------------------------------------------------------- throughput
THROUGHPUT_TAG = "HDS:F17"      # hydrotreated diesel to shop 8
ENERGY_TAGS = ["AVT:T55", "HDS:T5", "AVT:F29"]   # proxies, see assumptions

# ---------------------------------------------------------------- severity
#: Proxies of eq 61 (§29).  Reference and scale are calibrated robustly from
#: the training window (never presented as industrial limits).
SEVERITY_PROXIES: list[dict] = [
    {"tag": "HDS:W10", "label": "R-202 differential pressure", "direction": +1,
     "note": "Bed fouling / pressure drop; declared and observed kind agree."},
    {"tag": "AVT:T55", "label": "AVT furnace P-3 outlet temperature",
     "direction": +1, "note": "Furnace tube-skin proxy (§29)."},
    {"tag": "HDS:T5", "label": "HDS reactor outlet temperature", "direction": +1,
     "note": "Severity of the hydrotreating regime; also the deactivation "
             "handle of eq 62."},
    {"tag": "HDS:F26", "label": "HDS unit loading", "direction": +1,
     "note": "Loading vs the historical operating band (§29)."},
    {"tag": "AVT:F29", "label": "K-2 steam rate", "direction": +1,
     "note": "Column loading proxy."},
]


def all_channels() -> list[Channel]:
    return MANIPULATED + MEASURED_DISTURBANCES


def channel(key: str) -> Channel | None:
    for c in all_channels():
        if c.key == key:
            return c
    return None


MANIPULATED_KEYS = [c.key for c in MANIPULATED]
DISTURBANCE_KEYS = [c.key for c in MEASURED_DISTURBANCES]
