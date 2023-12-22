% https://github.com/Ziqi-Yang/latex_assignment_template
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{assignment}[a latex class for editing your assignment]

\LoadClass[10pt,letterpaper]{article}

\RequirePackage[left=1.5cm,right=1.5cm,top=1.5cm,bottom=1cm]{geometry}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{listings}
\RequirePackage{tcolorbox}
\usepackage{amsmath}

\setlength{\textwidth}{6.5in}
\setlength{\textheight}{9.in}
\setlength{\oddsidemargin}{0in}
\setlength{\headheight}{0in}

% create a box to highlight word
\newtcbox{\hword}[1][orange]{on line, arc = 0.3mm,auto outer arc,colback = #1!12!, colframe = #1!60!,colupper = black, boxsep =0mm, left =0.3mm, right = 0.3mm, top =0.4mm, bottom = 0.4mm,boxrule =0.2mm, bottomrule =0.2mm, toprule = 0.2mm}

% default code style
\lstset{
    framerule=1pt,
    frame=tb,
    emphstyle={\small\ttfamily\bfseries\color{Orange}},
    numbers=left,
    numberstyle= \tiny\color{black},
    basicstyle = \small\ttfamily,
    keywordstyle    = \bfseries\color{BrickRed},
    identifierstyle = \bfseries\color{black},
    stringstyle     = \bfseries\color{ForestGreen},
    commentstyle    = \bfseries\color{Violet},
    breaklines      =   true,
    columns         =   fixed,
    basewidth       =   .5em,
    backgroundcolor=\color{Gray!5},
    tabsize=2,
    showspaces=false,
    showstringspaces=false,
}

\newcommand{\assignmentTitle}[9]{
    \begin{flushleft}
    {\textbf #1\\#2\\#3\\#4\\#5\\}
    \end{flushleft}

    \begin{flushright}\vspace{-5mm}
    \includegraphics[height=2cm]{#6}
    \end{flushright}
    
    \begin{center}\vspace{-1cm}
    \textbf{\large #7}\\
    \end{center}

    \begin{flushleft}
    \textbf{\large #8}\\
    \end{flushleft}

    \begin{flushleft}
    \textbf{\large #9}\\
    \end{flushleft}
    
    \noindent\rule{\linewidth}{.6pt}

}