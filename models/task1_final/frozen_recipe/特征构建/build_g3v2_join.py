# -*- coding: utf-8 -*-
"""g3v2 IMU 成员特征直连：imu_vehicle.csv(扫描产物) + 协议标签 + imu_missing 标记。
用法：python build_g3v2_join.py --imu-csv <imu_vehicle.csv> --out <g3v2_features.csv>
窗口纪律：IMU 扫描表本身已按扫描窗口过滤，本脚本只做 join，不重新过滤。"""
import argparse, os, sys
import pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-csv", required=True)
    ap.add_argument("--protocol-dir", required=True, help="含 labels.csv/splits.csv 的受控目录")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pdir = a.protocol_dir
    labels = pd.read_csv(os.path.join(pdir, "labels.csv"), dtype={"sample_id": str})
    labels["gpsno"] = labels.sample_id.str.split("_").str[0]
    imu = pd.read_csv(a.imu_csv, dtype={"gpsno": str})
    f = labels[["sample_id", "gpsno"]].merge(imu, on="gpsno", how="left")
    f["imu_missing"] = f.imu_rows.isna().astype(int)
    f = f.drop(columns=["imu_rows"])
    f.to_csv(a.out, index=False, lineterminator="\n")
    print("saved", a.out, f.shape, "| imu_missing:", int(f.imu_missing.sum()))
if __name__ == "__main__":
    main()
