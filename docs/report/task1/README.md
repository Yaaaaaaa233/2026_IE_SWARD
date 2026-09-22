# 任务一提交物 LaTeX 框架（算法说明文档）

对照赛题《6.1 提交内容》：**预测结果 CSV + 算法说明文档（PDF）+ 可复现代码（含运行说明）**。
本目录是**说明文档**的 LaTeX 源码，全组共用一套框架，各自填充自己的章节。

## 如何编译

在本目录下执行（需 XeLaTeX，中文文档）：

```bash
xelatex main.tex    # 跑两遍，生成目录与交叉引用
xelatex main.tex
# 或：latexmk -xelatex main.tex
```

## 目录结构与分工

| 文件 | 章节 | 负责人 | 状态 |
| --- | --- | --- | --- |
| `main.tex` | 主文件（封面、目录、组装） | 组长 | ✅ 可用，封面队伍信息提交前统一替换 |
| `preamble.tex` | 宏包与样式（\TODO 黄框等） | 组长 | ✅ 一般不动；新增宏包在此登记 |
| `sections/sec01_overview.tex` | 第1章 赛题理解与总体技术路线 | 组长/全员 | 骨架，缺技术路线图 |
| `sections/sec02_data.tex` | 第2章 数据概览与版本管理 | 数据线 | ✅ 已按 2.0 版实测预填 |
| `sections/sec03_cleaning.tex` | 第3章 数据清洗与质量保障 | **数据线（我）** | ✅ **已完稿** |
| `sections/sec04_features.tex` | 第4章 特征工程 | 数据线 + 建模线 | 4.1 已预填；4.2–4.4 待填 |
| `sections/sec05_label_split.tex` | 第5章 标签、划分与评估 | 验证负责人 | 已预填数据侧承诺，待冻结补充 |
| `sections/sec06_model.tex` | 第6章 模型选择与参数设置 | 建模线 | 6.1 基线已预填；6.2/6.3 待填 |
| `sections/sec07_results.tex` | 第7章 实验结果与关键发现 | 建模线 | 7.1 基线已预填；7.2/7.3 待填 |
| `sections/sec08_conclusion.tex` | 第8章 总结与展望 | 组长/全员 | 骨架 |
| `sections/appx_code.tex` | 附录A 代码与复现说明 | 各代码负责人 | 清洗线脚本已列表，其余待补 |
| `sections/appx_dict.tex` | 附录B 特征字典节选 | 数据线 | ✅ 已预填 |
| `sections/appx_open.tex` | 附录C 官方待确认问题 | 组长 | ✅ 已按统一配置预填（11 项） |

## 团队约定

1. **只编辑自己负责的 `sections/*.tex` 文件**，不要改别人的；公共样式改动走 `preamble.tex` 并在群里说明。
2. 文中所有 **黄色【待填写】框（`\TODO{...}`）** 在最终提交前必须全部删除——提交前全局搜索 `TODO` 自查。
3. 表格统一用 `booktabs` 三线表 + `tabularx`（示例见第 3 章）；重点结论放 `hlbox`；原始数据样例放 `databox`。
4. 图片统一放 `figures/`，正文 `\includegraphics{文件名}` 即可（已设 graphicspath）。
5. 所有数字必须与 2.0 版数据产物对账，**禁止混用 1.0 版数字**；引用别人的实验数字前先问对方是否为冻结版。
6. 中文文档用 XeLaTeX 编译；文件名保持 ASCII（现有命名勿改），避免跨平台编译问题。
7. **脱敏红线**：本模板会同步到公开团队仓库，正文一律用代号车辆A/B 指代样例车
   （车辆A 即正类事故样本），坐标只保留两位小数加省略号；禁止写入真实设备号
   （gpsno/imei）、明细行或本机盘符路径。

## 依赖

TinyTeX / TeX Live 发行版，用到：ctex、booktabs、tabularx、longtable、tcolorbox、listings、enumitem、fancyhdr、hyperref 等（Windows 自带中文字体，`fontset=windows`）。
