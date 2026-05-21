% thesis.cls — IIT BHU IDD thesis class
% Manik Sharma, Roll 21174015, Dept of Physics
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[2026/05/19 IIT BHU IDD thesis]

\LoadClass[12pt,a4paper,oneside,openany]{book}

% Margins: 1.5in binding (inner), 1.0in elsewhere — IIT BHU guideline.
\RequirePackage[a4paper,left=1.5in,right=1in,top=1in,bottom=1in,headheight=15pt]{geometry}
\RequirePackage{mathptmx}        % Times-like serif
\RequirePackage[scaled=0.92]{helvet}
\RequirePackage{courier}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{setspace}
\onehalfspacing
\RequirePackage{graphicx}
\RequirePackage{amsmath,amssymb}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{placeins}
\RequirePackage{microtype}
\RequirePackage{xcolor}
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage{bibtopic}
\RequirePackage{tikz}
\usetikzlibrary{positioning,arrows.meta,calc,shapes.geometric,fit,backgrounds}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{enumitem}
\RequirePackage[hidelinks,colorlinks=true,
                linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}

% Captions
\captionsetup{font=small,labelfont=bf,labelsep=colon}

% Headers / footers — twoside layout, no personalised running foot.
\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\thepage}
\fancyhead[L]{\nouppercase\leftmark}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% Chapter heading: "Chapter N" line then bold title.
\titleformat{\chapter}[display]
  {\normalfont\Large\bfseries}{\chaptertitlename\ \thechapter}{12pt}{\LARGE}
\titlespacing*{\chapter}{0pt}{20pt}{20pt}

% Title-page macros
\newcommand{\thesistitle}[1]{\gdef\@thesistitle{#1}}
\newcommand{\thesissubtitle}[1]{\gdef\@thesissubtitle{#1}}
\newcommand{\authorname}[1]{\gdef\@authorname{#1}}
\newcommand{\rollnumber}[1]{\gdef\@rollnumber{#1}}
\newcommand{\degreename}[1]{\gdef\@degreename{#1}}
\newcommand{\branchname}[1]{\gdef\@branchname{#1}}
\newcommand{\departmentname}[1]{\gdef\@departmentname{#1}}
\newcommand{\institutename}[1]{\gdef\@institutename{#1}}
\newcommand{\supervisor}[1]{\gdef\@supervisor{#1}}
\newcommand{\cosupervisor}[1]{\gdef\@cosupervisor{#1}}
\newcommand{\submissionyear}[1]{\gdef\@submissionyear{#1}}
\newcommand{\submissionmonth}[1]{\gdef\@submissionmonth{#1}}

% User-level aliases — allow frontmatter files to access these without
% needing \makeatletter/\makeatother.
\newcommand{\thethesistitle}{\@thesistitle}
\newcommand{\thethesissubtitle}{\@thesissubtitle}
\newcommand{\theauthorname}{\@authorname}
\newcommand{\therollnumber}{\@rollnumber}
\newcommand{\thedegreename}{\@degreename}
\newcommand{\thebranchname}{\@branchname}
\newcommand{\thedepartmentname}{\@departmentname}
\newcommand{\theinstitutename}{\@institutename}
\newcommand{\thesupervisor}{\@supervisor}
\newcommand{\thecosupervisor}{\@cosupervisor}
\newcommand{\thesubmissionyear}{\@submissionyear}
\newcommand{\thesubmissionmonth}{\@submissionmonth}

% Defaults (override in main.tex)
\thesistitle{Joint-Embedding Predictive Pretraining for Solar Magnetic Winding-Flux Cubes}
\thesissubtitle{Toward Foundation-Model Pretraining for Solar Flare Forecasting}
\authorname{Manik Sharma}
\rollnumber{21174015}
\degreename{INTEGRATED DUAL DEGREE (B.TECH.\ + M.TECH.)}
\branchname{ENGINEERING PHYSICS}
\departmentname{DEPARTMENT OF PHYSICS}
\institutename{INDIAN INSTITUTE OF TECHNOLOGY\\(BANARAS HINDU UNIVERSITY)\\VARANASI -- 221 005}
\supervisor{Dr.~Abhishekh Kumar Srivastava}
\cosupervisor{}
\submissionmonth{May}
\submissionyear{2026}

% Title page (outer cover variant — cream stock per guidelines §3.2.1)
\newcommand{\maketitlepage}{%
  \begin{titlepage}
    \centering
    {\Large\bfseries \@thesistitle\par}
    \ifx\@thesissubtitle\empty\else
      \vspace{0.4cm}{\itshape \@thesissubtitle\par}\fi
    \vspace{1.0cm}
    \includegraphics[width=0.28\textwidth]{assets/iitbhu_logo.png}\par
    \vspace{0.8cm}
    {\normalsize Thesis submitted in partial fulfilment for the Award of\par}
    \vspace{0.3cm}
    {\normalsize\bfseries \@degreename\par}
    \vspace{0.2cm}
    {\normalsize in\par}
    \vspace{0.2cm}
    {\normalsize\bfseries \@branchname\par}
    \vspace{0.8cm}
    {\normalsize by\par}
    \vspace{0.4cm}
    {\large\bfseries \@authorname\par}
    \vspace{0.2cm}
    {\normalsize Roll No.~\@rollnumber\par}
    \vfill
    {\normalsize \@departmentname\par}
    {\normalsize \@institutename\par}
    \vspace{0.6cm}
    {\normalsize \@submissionmonth~\@submissionyear\par}
  \end{titlepage}
}

% Inner title page — adds supervisor block
\newcommand{\makeinnertitlepage}{%
  \begin{titlepage}
    \centering
    {\Large\bfseries \@thesistitle\par}
    \ifx\@thesissubtitle\empty\else
      \vspace{0.4cm}{\itshape \@thesissubtitle\par}\fi
    \vspace{1.0cm}
    \includegraphics[width=0.28\textwidth]{assets/iitbhu_logo.png}\par
    \vspace{0.6cm}
    {\normalsize Thesis submitted in partial fulfilment for the Award of\par}
    \vspace{0.2cm}
    {\normalsize\bfseries \@degreename\par}
    in {\normalsize\bfseries \@branchname\par}
    \vspace{0.6cm}
    {\normalsize by\par}
    {\large\bfseries \@authorname\par}
    {\normalsize Roll No.~\@rollnumber\par}
    \vspace{0.6cm}
    {\normalsize \textit{under the supervision of}\par}
    \vspace{0.2cm}
    {\normalsize\bfseries \@supervisor\par}
    \ifx\@cosupervisor\empty\else
      {\normalsize\bfseries \@cosupervisor\par}\fi
    \vfill
    {\normalsize \@departmentname\par}
    {\normalsize \@institutename\par}
    \vspace{0.4cm}
    {\normalsize \@submissionmonth~\@submissionyear\par}
  \end{titlepage}
}

\endinput
