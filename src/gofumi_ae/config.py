"""
定義ファイル（config/definition.json）の読み込みと検証ユーティリティ
仕様書 §15 「定義ファイル」に準拠
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_definition_file(definition_path: str) -> Dict[str, Any]:
    """
    定義ファイル（JSON）を読み込み、構造を検証して返す。
    
    Args:
        definition_path: config/definition.json のパス
        
    Returns:
        定義ファイルの辞書（version, feature_definition, category_definition 等を含む）
        
    Raises:
        FileNotFoundError: 定義ファイルが存在しない場合
        ValueError: JSON 解析エラーまたは必須フィールド不足の場合
    """
    if not os.path.isfile(definition_path):
        raise FileNotFoundError(f"Definition file not found: {definition_path}")
    
    try:
        with open(definition_path, "r", encoding="utf-8") as f:
            definition = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse definition JSON: {e}")
    
    # 必須フィールドの確認
    required_keys = [
        "version",
        "feature_definition",
        "category_definition",
        "missing_rules",
        "model_config",
        "threshold_config",
        "smoothing_config",
        "io_contract",
    ]
    missing = [k for k in required_keys if k not in definition]
    if missing:
        raise ValueError(f"Definition file missing required keys: {missing}")
    
    return definition


def extract_features_from_definition(definition: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Dict[str, Any]]]:
    """
    定義ファイルから特徴量情報を抽出する。
    
    Returns:
        (feature_order, categorical_features, feature_rules)
            - feature_order: 有効な特徴量のリスト（順序保持）
            - categorical_features: カテゴリ特徴量のリスト
            - feature_rules: 各特徴量の欠損・異常値ルール
    """
    feat_def = definition["feature_definition"]
    cat_def = definition["category_definition"]
    miss_def = definition["missing_rules"]
    
    # feature_order は定義ファイルで有効な特徴量を順序付けで提供
    feature_order = feat_def.get("feature_order", [])
    
    # enabled=true の特徴量のみを抽出（feature_order に含まれるもの）
    enabled_features = []
    feature_type_map = {}  # name -> type (continuous/categorical)
    feature_catalog = feat_def.get("feature_catalog", [])
    for feat_obj in feature_catalog:
        name = feat_obj.get("name")
        if feat_obj.get("enabled", False):
            enabled_features.append(name)
            feature_type_map[name] = feat_obj.get("type", "continuous")
    
    # feature_order に従って順序付け
    ordered_features = [f for f in feature_order if f in enabled_features]
    
    # カテゴリ特徴量の抽出
    categorical_features = [f for f in ordered_features if feature_type_map.get(f) == "categorical"]
    
    # 欠損ルール
    default_rule = miss_def.get("default_rule", {})
    rules_dict = miss_def.get("rules", {})
    feature_rules = {}
    for feat in ordered_features:
        if feat in rules_dict:
            feature_rules[feat] = rules_dict[feat]
        else:
            feature_rules[feat] = default_rule.copy()
    
    return ordered_features, categorical_features, feature_rules


def extract_category_maps_from_definition(definition: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    定義ファイルからカテゴリマッピング情報を抽出する。
    
    Returns:
        category_maps: {カテゴリ特徴量名: {カテゴリラベル: 数値コード}}
    """
    cat_def = definition["category_definition"]
    categories = cat_def.get("categories", {})
    
    # 有効なカテゴリのみを返す
    feat_def = definition["feature_definition"]
    enabled_categorical = set()
    for feat_obj in feat_def.get("feature_catalog", []):
        if feat_obj.get("enabled", False) and feat_obj.get("type") == "categorical":
            enabled_categorical.add(feat_obj.get("name"))
    
    return {k: v for k, v in categories.items() if k in enabled_categorical}


def extract_model_config_from_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    定義ファイルからモデル設定を抽出する（CLI オーバーライド不可の値のみ）。
    
    Returns:
        model_config: シンプルな値辞書（cli_overridable の値は除外）
    """
    model_cfg = definition["model_config"]
    
    # 基本パラメータを抽出（値のみ）
    config = {}
    for key in [
        "seq_len", "batch_size", "epochs", "lr", "d_model", "nhead", 
        "num_layers", "dim_ff", "dropout", "val_ratio", "weight_decay",
        "max_norm", "random_seed", "tail_steps", "selected_architecture"
    ]:
        if key in model_cfg:
            val = model_cfg[key]
            # cli_overridable 形式の場合は value を抽出、単純値の場合はそのまま
            config[key] = val.get("value") if isinstance(val, dict) else val
    
    return config


def extract_threshold_config_from_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    定義ファイルからしきい値設定を抽出する。
    
    Returns:
        threshold_config: percentile, threshold_source, score_target 等
    """
    thr_cfg = definition["threshold_config"]
    
    config = {}
    for key in ["percentile", "threshold_source", "score_target", "weight_policy"]:
        if key in thr_cfg:
            val = thr_cfg[key]
            config[key] = val.get("value") if isinstance(val, dict) else val
    
    # file_score_aggregation は value を抽出
    if "file_score_aggregation" in thr_cfg:
        val = thr_cfg["file_score_aggregation"]
        config["file_score_aggregation"] = val.get("value") if isinstance(val, dict) else val
    
    return config


def extract_smoothing_config_from_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    定義ファイルから平滑判定設定を抽出する。
    
    Returns:
        smoothing_config: smooth_window, threshold_coeff, consecutive_count 等
    """
    smooth_cfg = definition["smoothing_config"]
    
    config = {}
    for key in ["smooth_window", "threshold_coeff", "consecutive_count", "smoothing_method"]:
        if key in smooth_cfg:
            val = smooth_cfg[key]
            config[key] = val.get("value") if isinstance(val, dict) else val
    
    return config


def extract_io_contract_from_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    定義ファイルから入出力契約を抽出する。
    
    Returns:
        io_contract: input_format, encoding, time_column, output_columns, paths 等
    """
    io_cfg = definition["io_contract"]
    
    contract = {}
    contract["input_format"] = io_cfg.get("input_format", "csv")
    contract["encoding"] = io_cfg.get("encoding", "utf-8")
    contract["time_column"] = io_cfg.get("time_column", "datetime")
    contract["output_columns"] = io_cfg.get("output_columns", [])
    
    # paths から値のみを抽出（cli_overridable 除外）
    paths = {}
    for path_key, path_val in io_cfg.get("paths", {}).items():
        paths[path_key] = path_val.get("value") if isinstance(path_val, dict) else path_val
    
    contract["paths"] = paths
    
    return contract


def resolve_column_map_file(definition: Dict[str, Any], project_root: str) -> str:
    """
    定義ファイルから列名対応定義ファイルのパスを取得・検証する。
    
    Args:
        definition: 定義ファイル
        project_root: プロジェクトルートパス
        
    Returns:
        column_map.json の絶対パス
        
    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
    io_contract = extract_io_contract_from_definition(definition)
    column_map_path = io_contract["paths"].get("column_map_file", "config/column_map.json")
    
    # 相対パスの場合はプロジェクトルートを基準に解決
    if not os.path.isabs(column_map_path):
        column_map_path = os.path.join(project_root, column_map_path)
    
    if not os.path.isfile(column_map_path):
        raise FileNotFoundError(f"Column map file not found: {column_map_path}")
    
    return column_map_path


def load_column_map(column_map_path: str) -> Tuple[Dict[str, str], str]:
    """
    列名対応定義ファイルを読み込む。
    
    Returns:
        (column_map_dict, time_column)
            - column_map_dict: 特徴量名 -> CSV列名 のマッピング（空文字は特徴量名をそのまま使用）
            - time_column: 時刻列の CSV 列名
    """
    if not os.path.isfile(column_map_path):
        raise FileNotFoundError(f"Column map file not found: {column_map_path}")
    
    try:
        with open(column_map_path, "r", encoding="utf-8") as f:
            col_map_obj = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse column map JSON: {e}")
    
    column_map = col_map_obj.get("column_map", {})
    time_column = col_map_obj.get("time_column", "datetime")
    
    return column_map, time_column


def validate_definition_structure(definition: Dict[str, Any]) -> bool:
    """
    定義ファイルの構造を簡易検証する。
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    # 既に load_definition_file で必須フィールドはチェック済み
    # ここでは追加の構造検証を行う
    
    feat_def = definition.get("feature_definition", {})
    feature_order = feat_def.get("feature_order", [])
    if not feature_order:
        raise ValueError("feature_definition.feature_order is empty")
    
    return True


def extract_frame_range_config_from_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    定義ファイルからフレーム範囲設定を抽出する。
    
    Returns:
        frame_range_config: filter_column, start_value, end_value
            - filter_column: フィルタ対象列名（定義ファイルの frame_range_config から）
            - start_value: 範囲の下限（定義ファイルの frame_range_config から）
            - end_value: 範囲の上限（定義ファイルの frame_range_config から）
    """
    frame_cfg = definition.get("frame_range_config", {})
    
    config = {
        "filter_column": frame_cfg.get("filter_column"),
        "start_value": frame_cfg.get("start_value"),
        "end_value": frame_cfg.get("end_value"),
    }
    
    return config
