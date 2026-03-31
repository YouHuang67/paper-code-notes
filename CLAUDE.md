# Project: Paper & Code Notes

基于 MkDocs 的个人知识库，专注于论文阅读和代码分析。

## 论文阅读

### 流程
1. PDF 放入 `refs/papers/`，按修改时间自动识别最新文件
2. 预处理：`python scripts/split_pdf.py`
   - 在 PDF 同目录生成同名子目录 `refs/papers/<stem>/`
   - 内含：`meta.json`（元数据）、`full_text.txt`（纯文本）、`part_*.pdf`（拆分 PDF）
3. 用户提供 arXiv 地址
4. 笔记输出到 `docs/paper_reading/`，单篇论文一个 md 文件
- **定位论文素材**：当用户说"更新了一篇论文"时，直接按 `refs/papers/` 下修改时间最新的目录定位，读取其中的 `full_text.txt` 和 `part_*.pdf`，**禁止遍历所有论文目录**

### 笔记规范
- **开头**：论文链接 + 代码链接 + 团队。然后是「概述」章节，具体描述论文解决什么问题、怎么解决的、整体流程和主要结论，读完概述即可掌握全貌
- **正文**：按论文顺序展开，技术细节不省略（数学公式、训练配置、数据规模等），但行文简洁不啰嗦
- **格式**：纯文本 + 分点罗列为主，表格仅用于实验结果对比，不用 admonition 等花哨语法
- **末尾**：关键启示，提炼可迁移的设计思路

## 代码分析

### 流程
- 本地路径或 GitHub 地址由用户指定
- 笔记位置：`docs/code_analysis/<project>/`
- 文件命名：`00_overview.md`, `01_xxx.md`, `02_xxx.md`...
- 复杂项目拆多个文档按概念组织，简单项目单文件即可

### 源码引用格式
```markdown
**源码位置**: [函数名](https://github.com/org/repo/blob/main/path/file.py#L10-L25)
```

### 代码摘录规范（遵循全局 CLAUDE.md §7.3）
- 保持代码完整连贯，禁止 Phase/Step 标记打断
- 大段注释置于关键逻辑前（循环、算法阶段）
- 行尾精简注释：shape + 简要语义

```python
'''
Q 加载：make_block_ptr 构建指针，reshape 适配 tl.dot
'''
p_q = tl.make_block_ptr(...)
b_q = tl.load(p_q, boundary_check=(0, 1))  # [G, BK]
b_q = (b_q * scale).to(b_q.dtype)

'''
Online softmax 迭代：累积 max/sum/output
'''
for i_c in range(0, NC, BC):
    b_k = tl.load(p_k)                      # [BK, BC]
    b_s = tl.dot(b_q, b_k)                  # [G, BC]
    b_m = tl.maximum(b_m, tl.max(b_s, 1))   # [G] 更新 max
    ...
```

## 通用文档规范

- 格式简洁直接：纯 Markdown + 分点罗列，不用 admonition / tabs 等扩展语法
- 表格仅用于实验数据对比，其他场景用分点罗列
- 详略分明：开头概述让人快速了解全貌，正文按原始顺序详细展开不省略关键细节
- **禁止在阅读论文和分析代码过程中直接访问网址**（WebFetch / WebSearch 等），所有素材必须来自本地文件（PDF、源码、用户提供的信息）
- **禁止大量一次性写入文档**，必须分段分点逐步写入，避免大流量中断导致内容丢失

## 分类与检索

### Index 分类
- `paper_reading/index.md`：按研究主题分类（如 视频生成 / Sparse Attention）
- `code_analysis/index.md`：按技术方向分类（如 GPU Kernel / Model Architecture）
- 新增论文或代码分析时同步更新对应 index

### Tag 规范
- 用途：技术关键词，跨论文和代码通用，方便检索
- 格式：**大写字母单词，空格分隔**，禁止黏连（如 `Flash Attention` 而非 `FlashAttention`）
- 每个文档可有多个 tag
- 新增 tag 需更新下方列表并与用户确认

### 当前 Tag 列表
- `Triton` / `Sparse Attention` / `Flash Attention` / `Sliding Window`
- `Online Softmax` / `Bitonic Sort`
- `Video Generation` / `Diffusion Model` / `Diffusion Forcing` / `Flow Matching`
- `Reinforcement Learning`
- `LLM Inference`

### 索引维护：新增/修改文档时的完整流程

每次新增或修改文档，必须同步更新以下 3 处，保持一致：

1. **文档 frontmatter**（md 文件头部）
   ```yaml
   ---
   tags:
     - Triton
     - Sparse Attention
   ---
   ```
   - tags 必须从「当前 Tag 列表」中选取
   - 如需新增 tag：先与用户确认，确认后追加到本文件「当前 Tag 列表」

2. **`docs/data/tag_index.json`**（搜索索引数据）
   - 新增文档：追加一条记录
   - 修改文档：更新对应记录的 tags / summary
   - 字段：`title`、`path`（相对于 docs/ 的路径）、`type`（论文阅读/代码分析）、`tags`（数组，与 frontmatter 一致）、`summary`（一句话描述）
   - 此文件被 `docs/javascripts/tag_search.js` 读取，驱动标签页的实时搜索

3. **对应的 index.md 分类列表**
   - 论文 → `docs/paper_reading/index.md`（按研究主题分类）
   - 代码 → `docs/code_analysis/index.md`（按技术方向分类）

**检查项**：三处的 tag 列表必须完全一致（名称、大小写、数量）

## Git 规范

- Commit message：**简短英文小写**，一句话概括改动，不超过 50 字符
- 格式：`<type>: <description>`，如 `init: mkdocs knowledge base`、`add: skyreels-v2 paper notes`、`fix: tag index path`
- 禁止出现 AI 工具标识（Co-Authored-By 等）
- type 常用：`init` / `add` / `update` / `fix` / `refactor` / `docs`

## 前端缓存

- `mkdocs.yml` 中自定义 JS/CSS 文件带 `?v=N` 版本号参数
- **修改 JS 或 CSS 文件后，必须递增对应的版本号**（如 `?v=2` → `?v=3`），否则浏览器会加载旧缓存

## 目录结构

```
paper-code-notes/
├── docs/                    # ✅ 入库
│   ├── paper_reading/
│   └── code_analysis/
├── scripts/                 # ✅ 入库（工具脚本）
├── refs/                    # ❌ 不入库
│   └── papers/
└── mkdocs.yml
```
