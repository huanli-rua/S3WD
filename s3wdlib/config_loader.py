
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, yaml

GROUPS = {"DATA","LEVEL","KWB","S3WD","PSO"}

def _normalize_flat_to_grouped(raw: dict) -> dict:
    # flat -> grouped mapping (minimal set)
    D = {}

    # DATA
    dp = raw.get("DATA_PATH")
    if dp:
        ddir, dfile = os.path.split(dp)
    else:
        ddir, dfile = raw.get("DATA_DIR"), raw.get("DATA_FILE")
    data = {
        "data_dir": ddir or ".",
        "data_file": dfile,
        "continuous_label": raw.get("CONT_LABEL"),
        "threshold": raw.get("CONT_THRESH"),
        "threshold_op": raw.get("CONT_OP"),
        "label_col": raw.get("LABEL_COL"),
        "positive_label": raw.get("POSITIVE_LABEL"),
        "test_size": raw.get("TEST_SIZE"),
        "val_size": raw.get("VAL_SIZE"),
        "random_state": raw.get("SEED"),
    }
    D["DATA"] = data

    # LEVEL
    D["LEVEL"] = {
        "level_pcts": raw.get("LEVEL_PCTS"),
        "ranker": raw.get("RANKER"),
    }

    # KWB
    D["KWB"] = {
        "k": raw.get("KWB_K"),
        "metric": raw.get("KWB_metric","euclidean"),
        "eps": raw.get("KWB_eps", 1e-6),
    }

    # S3WD
    pen = raw.get("S3_penalty_large", raw.get("S3_pentalty_large"))
    D["S3WD"] = {
        "c1": raw.get("S3_c1"),
        "c2": raw.get("S3_c2"),
        "xi_min": raw.get("S3_xi_min"),
        "theta_pos": raw.get("S3_theta_pos"),
        "theta_neg": raw.get("S3_theta_neg"),
        "penalty_large": pen,
        "gamma_last": raw.get("S3_gamma_last", True),
        "gap": raw.get("S3_gap", 0.02),
    }

    # PSO
    D["PSO"] = {
        "particles": raw.get("PSO_particles"),
        "iters": raw.get("PSO_iters"),
        "w_max": raw.get("PSO_w_max"),
        "w_min": raw.get("PSO_w_min"),
        "c1": raw.get("PSO_c1"),
        "c2": raw.get("PSO_c2"),
        "seed": raw.get("PSO_seed"),
    }
    return D

def _require(group: dict, gname: str, keys):
    for k in keys:
        if k not in group:
            raise KeyError(f"{gname} 缺少字段: {k}")
        if group[k] is None:
            raise KeyError(f"{gname}.{k} 不能为空")

def load_yaml_cfg(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"YAML 为空或结构不是字典: {path}")

    # grouped or flat -> grouped
    if GROUPS & set(raw.keys()):
        cfg = raw
    else:
        cfg = _normalize_flat_to_grouped(raw)

    missing = [g for g in GROUPS if g not in cfg]
    if missing:
        raise KeyError(f"YAML 缺少分组: {missing}，必须包含 {sorted(GROUPS)}")

    # strict required keys
    _require(cfg["DATA"], "DATA", ["data_dir","data_file","test_size","val_size","random_state"])
    # Either continuous or label_col
    has_cont = all(cfg["DATA"].get(x) is not None for x in ["continuous_label","threshold","threshold_op"])
    has_label = all(cfg["DATA"].get(x) is not None for x in ["label_col","positive_label"])
    if not (has_cont or has_label):
        raise KeyError("DATA 需满足：连续标签 {continuous_label,threshold,threshold_op} 或 现成标签 {label_col,positive_label} 二选一")

    _require(cfg["LEVEL"], "LEVEL", ["level_pcts","ranker"])
    _require(cfg["KWB"],   "KWB",   ["k"])
    _require(cfg["S3WD"],  "S3WD",  ["c1","c2","xi_min","theta_pos","theta_neg","penalty_large","gamma_last"])
    _require(cfg["PSO"],   "PSO",   ["particles","iters","w_max","w_min","c1","c2","seed"])

    return cfg

def extract_vars(cfg: dict) -> dict:
    # 为已有代码提供扁平访问快捷键（不引入默认）
    V = {}
    D = cfg["DATA"]
    V["DATA_PATH"] = os.path.join(D["data_dir"], D["data_file"])
    if D.get("continuous_label") is not None:
        V["CONT_LABEL"] = D["continuous_label"]
        V["CONT_THRESH"] = D["threshold"]
        V["CONT_OP"] = D["threshold_op"]
    if D.get("label_col") is not None:
        V["LABEL_COL"] = D["label_col"]
        V["POSITIVE_LABEL"] = D["positive_label"]
    V["TEST_SIZE"] = D["test_size"]
    V["VAL_SIZE"] = D["val_size"]
    V["SEED"] = D["random_state"]

    L = cfg["LEVEL"]; V["LEVEL_PCTS"] = L["level_pcts"]; V["RANKER"] = L["ranker"]
    K = cfg["KWB"];   V["KWB_K"] = K["k"]
    S = cfg["S3WD"]
    V["S3_c1"]=S["c1"]; V["S3_c2"]=S["c2"]; V["S3_xi_min"]=S["xi_min"]
    V["S3_theta_pos"]=S["theta_pos"]; V["S3_theta_neg"]=S["theta_neg"]
    V["S3_penalty_large"]=S["penalty_large"]; V["S3_gamma_last"]=S["gamma_last"]; V["S3_gap"]=S.get("gap",0.02)
    P = cfg["PSO"]
    V["PSO_particles"]=P["particles"]; V["PSO_iters"]=P["iters"]
    V["PSO_w_max"]=P["w_max"]; V["PSO_w_min"]=P["w_min"]
    V["PSO_c1"]=P["c1"]; V["PSO_c2"]=P["c2"]; V["PSO_seed"]=P["seed"]
    return V

def show_cfg(cfg: dict) -> None:
    print("【配置快照】")
    for grp in ["DATA","LEVEL","KWB","S3WD","PSO"]:
        if grp in cfg:
            print(f"- {grp}: {cfg[grp]}")
