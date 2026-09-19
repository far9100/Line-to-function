# line2func

**Turn the lines in an image into mathematical functions.**<br>
**把圖片裡的線條變成數學函數。**

[English](#english) · [繁體中文](#繁體中文)

---

## English

line2func traces every line of a drawing into cubic curves and writes them as
equations you can paste into [Desmos](https://www.desmos.com/calculator):
parametric curves `x(t), y(t)` or explicit functions `y = f(x)` / `x = g(y)`.
It also saves SVG and LaTeX, and shows every curve and its equation in a local
web page. Everything runs on your own computer.

```
x(t) = −40.00·t³ +  60.00·t² +  60.00·t + 10.00
y(t) =  20.00·t³ − 270.00·t² + 250.00·t + 10.00        0 ≤ t ≤ 1
```

*(An arch from (10, 10) through (50, 70) to (90, 10), y axis up.)*

### Install

Python 3.10 or newer:

```bash
git clone https://github.com/far9100/Line-to-function.git line2func
cd line2func
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e .
```

Photos (the pretrained line-art network) and the neural engine also need
PyTorch; install a build for your GPU first, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu130` for an
NVIDIA RTX 50-series card (CPU only: `pip install torch`), then
`pip install -e ".[train,dev]"`.

### Use it in the browser

```bash
python -m line2func
```

The page opens in a new tab of your default browser.

1. **Drop a line drawing** onto the page, click **Choose a file…**, or paste
   with Ctrl+V.
2. Choose how the lines are written, as **functions** or as **parametric
   equations**, and how strongly noise is removed (**Remove noise**, 0-100; 50
   is the default, lower keeps more detail). Click **Convert**.
3. The result opens in the same page. Hover or click a curve to see its
   equations. Show the lines alone, over the original, or with the missed
   detail highlighted. Download SVG, JSON, Desmos, LaTeX or a ZIP, or click
   **Copy all for Desmos**.
4. **Clear image** starts over. Ctrl+C in the terminal, or **Quit** on the
   page, ends the program.

### Use it from the command line

```bash
python -m line2func.demo drawing.png --out out/                   # trace and write every output
python -m line2func.demo drawing.png --form function --out out/   # the equations as functions
python -m line2func.serve out/                                    # look at the result in the browser
```

| Output | Contents |
|---|---|
| `desmos.txt` | One Desmos equation per line |
| `equations.tex` | The same equations for LaTeX |
| `out.svg` | The drawing as a vector image |
| `curves.json` | Every curve: control points, stroke, width and color |
| `overlay.png` | The curves over the original, for a quick check |

| Option | Meaning |
|---|---|
| `--form {parametric,named,function}` | How the equations are written: `x(t), y(t)`, named lines and arcs, or `y = f(x)` / `x = g(y)` |
| `--curves N` | At most N curves (default 5,000) |
| `--denoise 0..100` | How strongly specks and short faint pieces are dropped (default 50) |
| `--threshold 0..1` | Ink threshold: lower it if faint lines are missed |
| `--lineart METHOD` | For photos: `informative`, `canny` or `xdog` |
| `--quality` | Also judge the result: `quality.png` marks missed detail |

`python -m line2func.demo --help` lists every option.

### Paste into Desmos

Open `desmos.txt`, select all and copy. Click the first expression box in
[Desmos](https://www.desmos.com/calculator) and paste: every line becomes one
expression. line2func makes at most 5,000 curves; if Desmos gets slow on your
computer, ask for fewer, e.g. `--curves 2000`.

### Good inputs

Clean line art works best: sketches, ink drawings, comics and scans, with dark
lines on light paper (light lines on dark paper are inverted automatically).
Photos need line extraction first: `--lineart informative` (after
`python -m line2func.weights fetch informative`), or `canny` / `xdog`.

### More

[docs/details.md](docs/details.md) is the full manual: every output and
option, tips with measurements, the neural engine, evaluation, using
line2func from Python, and the project layout.

### License

[GNU General Public License v3.0 or later](LICENSE). Third-party code and
weights: [docs/third_party.md](docs/third_party.md).

---

## 繁體中文

line2func 會把圖裡的每一條線描成三次曲線，並寫成可以貼進 [Desmos](https://www.desmos.com/calculator) 的算式：參數式 `x(t)`、`y(t)`，或顯函數 `y = f(x)`／`x = g(y)`。它也能輸出 SVG 和 LaTeX，並在本機網頁上顯示每條曲線與它的算式。全部在你自己的電腦上執行。

```
x(t) = −40.00·t³ +  60.00·t² +  60.00·t + 10.00
y(t) =  20.00·t³ − 270.00·t² + 250.00·t + 10.00        0 ≤ t ≤ 1
```

*（一道拱形：從 (10, 10) 經過 (50, 70) 到 (90, 10)，y 軸朝上。）*

### 安裝

需要 Python 3.10 以上：

```bash
git clone https://github.com/far9100/Line-to-function.git line2func
cd line2func
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e .
```

照片（預訓練線稿模型）和神經網路引擎還需要 PyTorch：先安裝符合你 GPU 的版本，例如 NVIDIA RTX 50 系列用 `pip install torch --index-url https://download.pytorch.org/whl/cu130`（只用 CPU：`pip install torch`），再執行 `pip install -e ".[train,dev]"`。

### 用瀏覽器

```bash
python -m line2func
```

會在預設瀏覽器開一個新分頁。

1. **把線稿拖進頁面**，或按〔選擇檔案…〕，或按 Ctrl+V 貼上。
2. 選擇線段寫成**函數**或**參數方程式**，以及去雜訊的強度（〔去雜訊〕，0–100；預設 50，調低保留更多細節），再按〔確認 ▶〕。
3. 結果會在同一頁開啟。滑過或點擊曲線可以看它的算式；可以只顯示線段、疊在原圖上，或標出遺漏的線段；也可以下載 SVG、JSON、Desmos、LaTeX 或 ZIP，或按〔全部複製到 Desmos〕。
4. 〔清除圖片〕會重新開始。在終端機按 Ctrl+C，或按頁面上的〔結束〕，就會結束程式。

### 用指令列

```bash
python -m line2func.demo drawing.png --out out/                   # 描線並寫出所有輸出檔
python -m line2func.demo drawing.png --form function --out out/   # 算式寫成函數
python -m line2func.serve out/                                    # 在瀏覽器查看結果
```

| 輸出檔 | 內容 |
|---|---|
| `desmos.txt` | 每行一個 Desmos 算式 |
| `equations.tex` | 同樣的算式，給 LaTeX 用 |
| `out.svg` | 向量圖 |
| `curves.json` | 每條曲線的控制點、所屬筆畫、線寬與顏色 |
| `overlay.png` | 曲線疊在原圖上，方便快速檢查 |

| 選項 | 意義 |
|---|---|
| `--form {parametric,named,function}` | 算式的寫法：`x(t)`、`y(t)`，具名的直線與圓弧，或 `y = f(x)`／`x = g(y)` |
| `--curves N` | 最多 N 條曲線（預設 5,000） |
| `--denoise 0..100` | 小雜點和短的淡線片段丟掉的強度（預設 50） |
| `--threshold 0..1` | 墨跡門檻：淡的線被漏掉時調低 |
| `--lineart 方法` | 照片用：`informative`、`canny` 或 `xdog` |
| `--quality` | 另外評估結果：`quality.png` 會標出遺漏的細節 |

`python -m line2func.demo --help` 會列出所有選項。

### 貼進 Desmos

打開 `desmos.txt`，全選並複製，點 [Desmos](https://www.desmos.com/calculator) 的第一個算式欄並貼上，每一行會成為一個算式。line2func 最多產生 5,000 條曲線；如果在你的電腦上 Desmos 變慢，可以指定少一點，例如 `--curves 2000`。

### 適合的輸入

乾淨的線稿效果最好：素描、墨線稿、漫畫、掃描稿，淺色紙上的深色線條（深色背景上的淺色線條會自動反轉）。照片要先抽線稿：`--lineart informative`（先執行 `python -m line2func.weights fetch informative`），或 `canny`／`xdog`。

### 更多說明

[docs/details.md](docs/details.md#繁體中文) 是完整說明：所有輸出與選項、附實測數據的訣竅、神經網路引擎、評估方式、在 Python 中使用，以及專案結構。

### 授權

[GNU General Public License v3.0 或更新版本](LICENSE)。第三方程式碼與權重：[docs/third_party.md](docs/third_party.md)。
