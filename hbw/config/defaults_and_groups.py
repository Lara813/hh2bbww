# coding: utf-8

import law

from columnflow.inference import InferenceModel
from columnflow.tasks.framework.base import RESOLVE_DEFAULT


def default_calibrator(container):
    return "with_b_reg"


def default_selector(container):
    if container.has_tag("is_sl"):
        selector = "sl1"
    elif container.has_tag("is_dl"):
        selector = "dl1"

    return selector


def ml_inputs_producer(container):
    if container.has_tag("is_sl") and not container.has_tag("is_resonant"):
        ml_inputs = "sl_ml_inputs"
    if container.has_tag("is_dl"):
        ml_inputs = "dl_ml_inputs"
    if container.has_tag("is_sl") and container.has_tag("is_resonant"):
        ml_inputs = "sl_res_ml_inputs"
    return ml_inputs


def default_ml_model(cls, container, task_params):
    """ Function that chooses the default_ml_model based on the inference_model if given """
    # for most tasks, do not use any default ml model
    default_ml_model = None

    # set default ml_model when task is part of the MLTraining pipeline
    # NOTE: default_ml_model does not work for the MLTraining task
    if hasattr(cls, "ml_model"):
        # TODO: we might want to distinguish between multiple default ML models (sl vs dl)
        default_ml_model = "dense_default"

    # check if task is using an inference model
    # if that is the case, use the default ml_model set in the inference model
    if getattr(cls, "inference_model", None):
        inference_model = task_params.get("inference_model", None)

        # if inference model is not set, assume it's the container default
        if inference_model in (None, law.NO_STR, RESOLVE_DEFAULT):
            inference_model = container.x.default_inference_model

        # get the default_ml_model from the inference_model_inst
        inference_model_inst = InferenceModel.get_cls(inference_model)
        default_ml_model = getattr(inference_model_inst, "ml_model_name", default_ml_model)

    return default_ml_model


def default_producers(cls, container, task_params):
    """ Default producers chosen based on the Inference model and the ML Model """

    # per default, use the ml_inputs and event_weights
    default_producers = [ml_inputs_producer(container), "event_weights", "pre_ml_cats"]

    if hasattr(cls, "ml_model"):
        # do no further resolve the ML categorizer when this task is part of the MLTraining pipeline
        default_producers.remove("pre_ml_cats")
        return default_producers

    # check if a mlmodel has been set
    ml_model = task_params.get("ml_models", None)

    # only consider 1 ml_model
    if ml_model and isinstance(ml_model, (list, tuple)):
        ml_model = ml_model[0]

    # try and get the default ml model if not set
    if ml_model in (None, law.NO_STR, RESOLVE_DEFAULT):
        ml_model = default_ml_model(cls, container, task_params)

    # if a ML model is set, and the task is not part of the MLTraining pipeline,
    # use the ml categorization producer instead of the default categorization producer
    if ml_model not in (None, law.NO_STR, RESOLVE_DEFAULT, tuple()):
        default_producers.remove("pre_ml_cats")
        # NOTE: this producer needs to be added as the last element! otherwise, category_ids will be overwritten
        default_producers.append(f"cats_ml_{ml_model}")

    return default_producers


def set_config_defaults_and_groups(config_inst):
    """ Configuration function that sets all the defaults and groups in the config_inst """
    year = config_inst.campaign.x.year

    # define the default dataset and process based on the analysis tags
    signal_tag = "sl" if config_inst.has_tag("is_sl") else "dl"
    default_signal_process = f"ggHH_kl_1_kt_1_{signal_tag}_hbbhww"
    signal_generator = "powheg"

    if config_inst.has_tag("resonant"):
        signal_tag = f"res_{signal_tag}"
        # for resonant, rely on the law.cfg to define the default signal process (NOTE: might change in the future)
        default_signal_process = law.config.get_expanded("analysis", "default_dataset")
        signal_generator = "madgraph"

    #
    # Defaults
    #

    # TODO: the default dataset is currently still being set up by the law.cfg
    config_inst.x.default_dataset = default_signal_dataset = f"{default_signal_process}_{signal_generator}"
    config_inst.x.default_calibrator = default_calibrator(config_inst)
    config_inst.x.default_selector = default_selector(config_inst)
    config_inst.x.ml_inputs_producer = ml_inputs_producer(config_inst)
    config_inst.x.default_producer = default_producers
    config_inst.x.default_ml_model = default_ml_model
    config_inst.x.default_inference_model = "default" if year == 2017 else "sl_22"
    config_inst.x.default_categories = ["incl"]
    config_inst.x.default_variables = ["jet1_pt"]

    #
    # Groups
    #

    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    config_inst.x.process_groups = {
        "all": ["*"],
        "default": [default_signal_process, "tt", "st", "w_lnu", "dy_lep"],
        "with_qcd": [default_signal_process, "tt", "qcd", "st", "w_lnu", "dy_lep"],
        "much": [default_signal_process, "tt", "qcd_mu", "st", "w_lnu", "dy_lep"],
        "ech": [default_signal_process, "tt", "qcd_ele", "st", "w_lnu", "dy_lep"],
        "inference": ["ggHH_*", "tt", "st", "w_lnu", "dy_lep", "qcd_*"],
        "k2v": ["qqHH_*", "tt", "st", "w_lnu", "dy_lep", "qcd_*"],
        "ml": [default_signal_process, "tt", "st", "w_lnu", "dy_lep"],
        "ml_test": [default_signal_process, "st", "w_lnu"],
        "mldl": ["ggHH_kl_1_kt_1_dl_hbbhww", "tt", "st", "dy_lep"],
        "mlsl": ["ggHH_kl_1_kt_1_sl_hbbhww", "tt", "st", "w_lnu", "dy_lep"],
        "test": [default_signal_process, "tt_sl"],
        "small": [default_signal_process, "tt", "st"],
        "bkg": ["tt", "st", "w_lnu", "dy_lep"],
        "signal": ["ggHH_*", "qqHH_*"], "gghh": ["ggHH_*"], "qqhh": ["qqHH_*"],
    }
    config_inst.x.process_groups["dmuch"] = ["data_mu"] + config_inst.x.process_groups["much"]
    config_inst.x.process_groups["dech"] = ["data_e", "data_egamma"] + config_inst.x.process_groups["ech"]

    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    config_inst.x.dataset_groups = {
        "all": ["*"],
        "default": [default_signal_dataset, "tt_*", "qcd_*", "st_*", "dy_*", "w_lnu_*"],
        "inference": ["ggHH_*", "tt_*", "qcd_*", "st_*", "dy_*", "w_lnu_*"],
        "test": [default_signal_dataset, "tt_sl_powheg"],
        "small": [default_signal_dataset, "tt_*", "st_*"],
        "bkg": ["tt_*", "st_*", "w_lnu_*", "dy_*"],
        "tt": ["tt_*"], "st": ["st_*"], "w": ["w_lnu_*"], "dy": ["dy_*"],
        "qcd": ["qcd_*"], "qcd_mu": ["qcd_mu*"], "qcd_ele": ["qcd_em*", "qcd_bctoe*"],
        "signal": ["ggHH_*", "qqHH_*"], "gghh": ["ggHH_*"], "qqhh": ["qqHH_*"],
        "ml": ["ggHH_kl_1*", "tt_*", "st_*", "dy_*", "w_lnu_*"],
        "dilep": ["tt_*", "st_*", "dy_*", "ggHH_*", "w_lnu_*"],
    }

    # category groups for conveniently looping over certain categories
    # (used during plotting and for rebinning)
    config_inst.x.category_groups = {
        "sl_much": ["sr__1mu", "sr__1mu__resolved", "sr__1mu__boosted"],
        "sl_ech": ["sr__1e", "sr__1e__resolved", "sr__1e__boosted"],
        "sl_much_resolved": ["sr__1mu__resolved", "sr__1mu__resolved__1b", "sr__1mu__resolved__2b"],
        "sl_ech_resolved": ["sr__1e__resolved", "sr__1e__resolved__1b", "sr__1e__resolved__2b"],
        "sl_much_boosted": ["sr__1mu__boosted"],
        "sl_ech_boosted": ["sr__1e__boosted"],
        "default": ["incl", "sr__1e", "sr__1mu"],
        "test": ["incl", "sr__1e"],
        "dilep": ["incl", "sr__2e", "sr__2mu", "sr__emu"],
        # Single lepton
        "SR_sl": (
            "sr__1e__1b__ml_ggHH_kl_1_kt_1_sl_hbbhww", "sr__1mu__1b__ml_ggHH_kl_1_kt_1_sl_hbbhww",
            "sr__1e__2b__ml_ggHH_kl_1_kt_1_sl_hbbhww", "sr__1mu__2b__ml_ggHH_kl_1_kt_1_sl_hbbhww",
        ),
        "vbfSR_sl": (
            "sr__1e__1b__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww", "sr__1mu__1b__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
            "sr__1e__2b__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww", "sr__1mu__2b__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
        ),
        "SR_sl_resolved": (
            "sr__1e__resolved__1b__ml_ggHH_kl_1_kt_1_sl_hbbhww", "sr__1mu__resolved__1b__ml_ggHH_kl_1_kt_1_sl_hbbhww",
            "sr__1e__resolved__2b__ml_ggHH_kl_1_kt_1_sl_hbbhww", "sr__1mu__resolved__2b__ml_ggHH_kl_1_kt_1_sl_hbbhww",
        ),
        "vbfSR_sl_resolved": (
            "sr__1e__resolved__1b__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
            "sr__1mu__resolved__1b__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
            "sr__1e__resolved__2b__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
            "sr__1mu__resolved__2b__ml_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
        ),
        "SR_sl_boosted": (
            "sr__1e__boosted__ml_ggHH_kl_1_kt_1_sl_hbbhww", "sr__1mu__boosted__ml_ggHH_kl_1_kt_1_sl_hbbhww",
        ),
        "vbfSR_sl_boosted": (
            "sr__1e__ml_boosted_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww", "sr__1mu__ml_boosted_qqHH_CV_1_C2V_1_kl_1_sl_hbbhww",
        ),
        "BR_sl": (
            "sr__1e__ml_tt", "sr__1e__ml_st", "sr__1e__ml_v_lep",
            "sr__1mu__ml_tt", "sr__1mu__ml_st", "sr__1mu__ml_v_lep",
        ),
        # Dilepton
        "SR_dl": (
            "sr__2e__1b__ml_ggHH_kl_1_kt_1_dl_hbbhww", "sr__2e__2b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
            "sr__2mu__1b__ml_ggHH_kl_1_kt_1_dl_hbbhww", "sr__2mu__2b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
            "sr__emu__1b__ml_ggHH_kl_1_kt_1_dl_hbbhww", "sr__emu__2b__ml_ggHH_kl_1_kt_1_dl_hbbhww",
        ),
        "BR_dl": (
            "sr__2e__ml_tt", "sr__2e__ml_st", "sr__2e__ml_dy_lep",
            "sr__2mu__ml_tt", "sr__2mu__ml_st", "sr__2mu__ml_dy_lep",
            "sr__emu__ml_tt", "sr__emu__ml_st", "sr__emu__ml_dy_lep",
        ),
    }

    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    config_inst.x.variable_groups = {
        "sl_resolved": ["n_*", "electron_*", "muon_*", "met_*", "jet*", "bjet*", "ht"],
        "sl_boosted": ["n_*", "electron_*", "muon_*", "met_*", "fatjet_*"],
        "default": ["n_jet", "n_muon", "n_electron", "ht", "m_bb", "deltaR_bb", "jet1_pt"],  # n_deepjet, ....
        "test": ["n_jet", "n_electron", "jet1_pt"],
        "cutflow": ["cf_jet1_pt", "cf_jet4_pt", "cf_n_jet", "cf_n_electron", "cf_n_muon"],  # cf_n_deepjet
        "dilep": [
            "n_jet", "n_muon", "n_electron", "ht", "m_bb", "m_ll", "deltaR_bb", "deltaR_ll",
            "ll_pt", "bb_pt", "E_miss", "delta_Phi", "MT", "min_dr_lljj",
            "m_lljjMET", "channel_id", "n_bjet", "wp_score", "m_ll_check",
        ],
        "control": [
            "n_jet", "n_fatjet", "n_electron", "n_muon",
            "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btagDeepFlavB",   # "jet1_btagDeepB",
            "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btagDeepFlavB",   # "jet2_btagDeepB",
            "jet3_pt", "jet3_eta", "jet3_phi", "jet3_btagDeepFlavB",   # "jet3_btagDeepB",
            "jet4_pt", "jet4_eta", "jet4_phi", "jet4_btagDeepFlavB",   # "jet4_btagDeepB",
            "fatjet1_pt", "fatjet1_eta", "fatjet1_phi", "fatjet1_btagHbb", "fatjet1_deepTagMD_HbbvsQCD",
            "fatjet1_mass", "fatjet1_msoftdrop", "fatjet1_tau1", "fatjet1_tau2", "fatjet1_tau21",
            "fatjet2_pt", "fatjet2_eta", "fatjet2_phi", "fatjet2_btagHbb", "fatjet2_deepTagMD_HbbvsQCD",
            "fatjet2_mass", "fatjet2_msoftdrop", "fatjet2_tau1", "fatjet2_tau2", "fatjet2_tau21",
            "electron_pt", "electron_eta", "electron_phi", "muon_pt", "muon_eta", "muon_phi",
        ],
    }

    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    config_inst.x.shift_groups = {
        "jer": ["nominal", "jer_up", "jer_down"],
    }

    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    # NOTE: this could be added as part of the selector init itself
    config_inst.x.selector_step_groups = {
        "resolved": ["Trigger", "Lepton", "VetoLepton", "Jet", "Bjet", "VetoTau"],
        "boosted": ["Trigger", "Lepton", "VetoLepton", "FatJet", "Boosted"],
        "default": ["Lepton", "VetoLepton", "Jet", "Bjet", "Trigger"],
        "thesis": ["Lepton", "Muon", "Jet", "Trigger", "Bjet"],  # reproduce master thesis cuts for checks
        "test": ["Lepton", "Jet", "Bjet"],
        "dilep": ["Jet", "Bjet", "Lepton", "Trigger"],
    }

    # plotting settings groups
    # (used in plotting)
    config_inst.x.general_settings_groups = {
        "test1": {"p1": True, "p2": 5, "p3": "text", "skip_legend": True},
        "default_norm": {"shape_norm": True, "yscale": "log"},
    }
    config_inst.x.process_settings_groups = {
        "default": {default_signal_process: {"scale": 2000, "unstack": True}},
        "unstack_all": {proc.name: {"unstack": True} for proc in config_inst.processes},
        "unstack_signal": {proc.name: {"unstack": True} for proc in config_inst.processes if "HH" in proc.name},
        "scale_signal": {
            proc.name: {"unstack": True, "scale": 10000}
            for proc, _, _ in config_inst.walk_processes() if proc.has_tag("is_signal")
        },
        "dilep": {
            "ggHH_kl_0_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
            "ggHH_kl_1_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
            "ggHH_kl_2p45_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
            "ggHH_kl_5_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
        },
        "dileptest": {
            "ggHH_kl_1_kt_1_dl_hbbhww": {"scale": 10000, "unstack": True},
        },
        "control": {
            "ggHH_kl_0_kt_1_sl_hbbhww": {"scale": 90000, "unstack": True},
            "ggHH_kl_1_kt_1_sl_hbbhww": {"scale": 90000, "unstack": True},
            "ggHH_kl_2p45_kt_1_sl_hbbhww": {"scale": 90000, "unstack": True},
            "ggHH_kl_5_kt_1_sl_hbbhww": {"scale": 90000, "unstack": True},
        },
    }
    # when drawing DY as a line, use a different type of yellow
    config_inst.x.process_settings_groups["unstack_all"].update({"dy_lep": {"unstack": True, "color": "#e6d800"}})

    config_inst.x.variable_settings_groups = {
        "test": {
            "mli_mbb": {"rebin": 2, "label": "test"},
            "mli_mjj": {"rebin": 2},
        },
    }

    # groups for custom plot styling
    config_inst.x.custom_style_config_groups = {
        "small_legend": {
            "legend_cfg": {"ncols": 2, "fontsize": 16},
        },
        "example": {
            "legend_cfg": {"title": "my custom legend title", "ncols": 2},
            "ax_cfg": {"ylabel": "my ylabel", "xlim": (0, 100)},
            "rax_cfg": {"ylabel": "some other ylabel"},
            "annotate_cfg": {"text": "category label usually here"},
        },
    }

    # CSP (calibrator, selector, producer) groups
    config_inst.x.producer_groups = {
        "mli": ["ml_inputs", "event_weights"],
        "mlo": ["ml_dense_default", "event_weights"],
        "cols": ["mli", "features"],
    }

    # groups are defined via config.x.category_groups
    config_inst.x.default_bins_per_category = {
        # Single lepton
        "SR_sl": 10,
        "vbfSR_sl": 5,
        "BR_sl": 3,
        "SR_sl_resolved": 10,
        "SR_sl_boosted": 5,
        "vbfSR_sl_resolved": 5,
        "vbfSR_sl_boosted": 3,
        # Dilepton
        "SR_dl": 10,
        "vbfSR_dl": 5,
        "BR_dl": 3,
        "SR_dl_resolved": 10,
        "SR_dl_boosted": 5,
        "vbfSR_dl_resolved": 5,
        "vbfSR_dl_boosted": 3,
    }

    config_inst.x.inference_category_rebin_processes = {
        # Single lepton
        "SR_sl": ("ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        "vbfSR_sl": ("ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        "SR_sl_resolved": ("ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        "SR_sl_boosted": ("ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        "vbfSR_sl_resolved": ("ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        "vbfSR_sl_boosted": ("ggHH_kl_1_kt_1_sl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_sl_hbbhww"),
        "BR_sl": lambda proc_name: "hbbhww" not in proc_name,
        # Dilepton
        "SR_dl": ("ggHH_kl_1_kt_1_dl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww"),
        "vbfSR_dl": ("ggHH_kl_1_kt_1_dl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww"),
        "SR_dl_resolved": ("ggHH_kl_1_kt_1_dl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww"),
        "SR_dl_boosted": ("ggHH_kl_1_kt_1_dl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww"),
        "vbfSR_dl_resolved": ("ggHH_kl_1_kt_1_dl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww"),
        "vbfSR_dl_boosted": ("ggHH_kl_1_kt_1_dl_hbbhww", "qqHH_CV_1_C2V_1_kl_1_dl_hbbhww"),
        "BR_dl": lambda proc_name: "hbbhww" not in proc_name,
    }
