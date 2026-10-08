\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hcmuitthesis}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ProcessOptions\relax
\LoadClass[14pt,a4paper,oneside]{memoir}

% ========= Config paper and font =============
\RequirePackage[T5]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage[english,vietnamese]{babel}
\RequirePackage{anyfontsize,mathptmx}

% ========== Require package to setup paper, link, language
\RequirePackage{geometry,graphicx,indentfirst,setspace}
\RequirePackage{tikz}
\RequirePackage[center]{caption}
\RequirePackage{fancyhdr,hyperref}
\RequirePackage{lipsum}
\RequirePackage{graphicx}

% ========== GLOSSARIES ==========
\RequirePackage[acronym,toc]{glossaries}
\setglossarystyle{long}

% ========== OTHER PACKAGES ==========
\RequirePackage[fontsize=13pt]{scrextend}
\RequirePackage{placeins}
\RequirePackage{amsfonts}
\RequirePackage{textcomp}
\RequirePackage{dblfloatfix}
\RequirePackage{lineno}
\RequirePackage{amssymb}
\RequirePackage{tabularx,booktabs}
\RequirePackage{longtable}
\RequirePackage{amsmath}
\RequirePackage{mathtools}
\RequirePackage{lscape}
\RequirePackage{caption}
\RequirePackage{lipsum,blindtext}
\RequirePackage{titlesec}
\RequirePackage{makecell}
\RequirePackage{listings}
\RequirePackage{subcaption}
\RequirePackage{adjustbox}
\RequirePackage{multirow}
\RequirePackage{rotating}

% ========== BIBLATEX CONFIGURATION ==========
% QUAN TRỌNG: Thêm csquotes TRƯỚC biblatex
\RequirePackage{csquotes}

\RequirePackage[
    backend=biber,
    style=ieee,
    sorting=nyt,
    maxbibnames=99,
    minbibnames=99,
    giveninits=true,
]{biblatex}

% Tùy chỉnh format cho IEEE
\DeclareNameAlias{author}{family-given}
\DeclareFieldFormat{title}{\mkbibquote{#1}}
\DeclareFieldFormat[article]{title}{\mkbibquote{#1}}

% Xử lý tách riêng tài liệu - DÙNG KEYWORDS thay vì LANGID
\DeclareSourcemap{
    \maps[datatype=bibtex]{
        \map{
            \step[fieldsource=keywords, match=vietnamese, final]
        }
        \map{
            \step[fieldsource=keywords, match=english, final]
        }
    }
}

% ========== TIKZ ==========
\usetikzlibrary{calc}

% ========== HYPERREF COLORS ==========
\hypersetup{
    urlcolor=blue,
    linkcolor=black,
    citecolor=black,
    colorlinks=true
}

% ========== DEFINE COLORS ==========
\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}

% ========== COLUMN TYPES ==========
\newcolumntype{C}[1]{>{\centering\arraybackslash}m{#1}}

% ========== CONFIG FIGURE PATH ==========
\graphicspath{{./imgs/}{./graphics/}{./images/}}

% ========== CHAPTER SPACING ==========
\newlength{\chaptertopmargin}
\setlength{\chaptertopmargin}{2cm}
\titlespacing*{\chapter}{0pt}{\chaptertopmargin}{20pt}

% ========== MATH CONFIG ==========
\DeclarePairedDelimiter\bra{\langle}{\rvert}
\DeclarePairedDelimiter\ket{\lvert}{\rangle}
\DeclarePairedDelimiterX\braket[2]{\langle}{\rangle}{#1 \delimsize\vert#2}

% ========== LISTINGS CONFIG ==========
\lstset{
  frame=tb,
  aboveskip=3mm,
  belowskip=3mm,
  showstringspaces=false,
  columns=fullflexible,
  basicstyle={\small\ttfamily},
  numbers=left,
  numberstyle=\tiny\color{gray},
  keywordstyle=\color{blue},
  commentstyle=\color{dkgreen},
  stringstyle=\color{mauve},
  breaklines=true,
  postbreak=\mbox{\textcolor{red}{$\hookrightarrow$}\space},
  tabsize=4
}

% ========== CUSTOM COMMANDS ==========
\newcommand{\@upperuniname}{}
\newcommand{\upperuniname}[1]{\renewcommand{\@upperuniname}{#1}}
\newcommand{\@uniname}{}
\newcommand{\uniname}[1]{\renewcommand{\@uniname}{#1}}
\newcommand{\@deptname}{}
\newcommand{\deptname}[1]{\renewcommand{\@deptname}{#1}}
\newcommand{\@reporttype}{ĐỒ ÁN CHUYÊN NGÀNH}
\newcommand{\reporttype}[1]{\renewcommand{\@reporttype}{#1}}
\newcommand{\@stumajor}{}
\newcommand{\stumajor}[1]{\renewcommand{\@stumajor}{#1}}
\newcommand{\@titleen}{}
\newcommand{\titleen}[1]{\renewcommand{\@titleen}{#1}}
\newcommand{\@defcouncil}{Council}
\newcommand{\defcouncil}[1]{\renewcommand{\@defcouncil}{#1}}
\newcommand{\@supervisor}{}
\newcommand{\supervisor}[1]{\renewcommand{\@supervisor}{#1}}
\newcommand{\@supervisorname}{}
\newcommand{\supervisorname}[1]{\renewcommand{\@supervisorname}{#1}}
\newcommand{\@secretary}{Secretary}
\newcommand{\secretary}[1]{\renewcommand{\@secretary}{#1}}
\newcommand{\@instruction}{}
\newcommand{\instruction}[1]{\renewcommand{\@instruction}{#1}}
\newcommand{\@stuname}{Trịnh Nguyễn Thanh Bình - 23520162\\
Bùi Quốc Bảo - 23520092\\
Diệp Thanh Phong - 23521160\\
Lê Trần Huỳnh Phong - 23521164}
\newcommand{\stuname}[1]{\renewcommand{\@stuname}{#1}}
\newcommand{\@stunamewithid}{}
\newcommand{\stunamewithid}[1]{\renewcommand{\@stunamewithid}{#1}}
\newcommand{\@reportplace}{}
\newcommand{\reportplace}[1]{\renewcommand{\@reportplace}{#1}}

% ========== OVERWRITE COMMANDS ==========
\renewcommand{\@title}{}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}
\renewcommand{\large}{\fontsize{15}{15}\selectfont}
\renewcommand{\Large}{\fontsize{20}{20}\selectfont}
\renewcommand{\arraystretch}{1.2}
\renewcommand*{\lstlistingname}{Đoạn mã}
\renewcommand\lstlistlistingname{Danh sách đoạn mã}

\renewcommand*{\glossarysection}[2][]{%
  \chapter*{Danh sách các từ viết tắt}
  \addcontentsline{toc}{chapter}{Danh sách các từ viết tắt}
}

% ========== PAPER MARGIN ==========
\geometry{
  paper=a4paper,
  left=3.5cm,
  right=2cm,
  top=3cm,
  bottom=3.5cm,
  includeheadfoot=true,
  headheight=30pt
}
\OnehalfSpacing
\raggedbottom

% ========== TOC DEPTH ==========
\setsecnumdepth{subsubsection}
\settocdepth{subsubsection}

% ========== TOC NUMWIDTH ==========
\setlength{\cfttablenumwidth}{3em}
\setlength{\cftfigurenumwidth}{3em}
\AtBeginDocument{\renewcommand*\bibname{References}}

% ========== HEADER/FOOTER ==========
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\footrulewidth}{0.4pt}
\renewcommand{\headrulewidth}{0pt}

\fancypagestyle{plain}{
    \fancyhf{}
    \renewcommand{\footrulewidth}{0.4pt}
    \fancyfoot[C]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
}

% ========== COVER PAGE ==========
\def\coverpage{
  \pagestyle{empty}
  \large\bfseries
  \begin{tikzpicture}[remember picture,overlay]
    \draw[line width=3pt](
    $(current page.north west) + (3.0cm,-2.0cm)$
    ) rectangle (
    $(current page.south east) + (-2.0cm,2.0cm)$
    );
    \draw[line width=1pt](
    $(current page.north west) + (2.9cm,-1.9cm)$
    ) rectangle (
    $(current page.south east) + (-1.9cm,1.9cm)$
    );
  \end{tikzpicture}
  \vspace*{-3\baselineskip}
  \begin{center}
    \uppercase{\@upperuniname} \\
    \uppercase{\@uniname} \\
    \uppercase{\@deptname} \\
    \vfill
    \uppercase{\@stuname}
    \vfill
    \uppercase{\@reporttype}
    \vfill
    {\Large{\parbox[c]{0.8\linewidth}{\centering\@title}}} \\
    \vfill
    {\Large{\parbox[c]{0.8\linewidth}{\centering\@titleen}}}
    \vfill
    \uppercase {\@stumajor}
  \end{center}
  \vfill
  \vfill
  \mdseries
  \begin{center}
    \uppercase{\@reportplace \the\year{}}
  \end{center}
  \normalsize
  \cleardoublepage
  \pagestyle{fancy}
}

% ========== SECOND COVER PAGE ==========
\def\secondcoverpage{
  \pagestyle{empty}
  \large\bfseries
  \begin{tikzpicture}[remember picture,overlay]
    \draw[line width=3pt](
    $(current page.north west) + (3.0cm,-2.0cm)$
    ) rectangle (
    $(current page.south east) + (-2.0cm,2.0cm)$
    );
    \draw[line width=1pt](
    $(current page.north west) + (2.9cm,-1.9cm)$
    ) rectangle (
    $(current page.south east) + (-1.9cm,1.9cm)$
    );
  \end{tikzpicture}
  \vspace*{-3\baselineskip}
  \begin{center}
    \uppercase{\@upperuniname} \\
    \uppercase{\@uniname} \\
    \uppercase{\@deptname} \\
    \vfill
    \uppercase{\@stunamewithid}
    \vfill
    \uppercase{\@reporttype}
    \vfill
    {\Large{\parbox[c]{0.8\linewidth}{\centering\@title}}} \\
    \vfill
    {\Large{\parbox[c]{0.8\linewidth}{\centering\@titleen}}}
    \vfill
    \uppercase {\@stumajor}
    \vfill
    \uppercase{\@supervisor} \\
    \uppercase{\@supervisorname}
  \end{center}
  \vfill
  \vfill
  \mdseries
  \begin{center}
    \uppercase{\@reportplace \the\year{}}
  \end{center}
  \normalsize
  \cleardoublepage
  \pagestyle{fancy}
}

% ========== MAIN THESIS ==========
\def\mainthesis{
  \counterwithin{equation}{chapter}
  \counterwithin{table}{chapter}
  \counterwithin{figure}{chapter}
  \setcounter{secnumdepth}{3}
  \mainmatter
  \pagestyle{fancy}
  \input{chapters/main/summary.tex}
}