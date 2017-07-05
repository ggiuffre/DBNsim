<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.0"
	xmlns:xhtml="http://www.w3.org/1999/xhtml"
	xmlns="http://www.w3.org/1999/xhtml"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
	xmlns:xs="http://www.w3.org/2001/XMLSchema"
	exclude-result-prefixes="xhtml xsl xs" >

<xsl:output method="text" version="1.0" encoding="UTF-8" />



<!-- MAIN TEMPLATE -->
<xsl:template match="/html">
<xsl:for-each select="//div[@id='content' and not(@class='no_tex')]">
<xsl:apply-templates />
</xsl:for-each>
</xsl:template>



<!-- h1 (section) -->
<xsl:template match="h1">
	\section{<xsl:apply-templates />}
</xsl:template>

<!-- h2 (subsection) -->
<xsl:template match="h2">
	\subsection{<xsl:apply-templates />}
</xsl:template>

<!-- h3 (subsubsection) -->
<xsl:template match="h3">
	\subsubsection{<xsl:apply-templates />}
</xsl:template>

<!-- h4 (paragraph) -->
<xsl:template match="h4">
	\paragraph{<xsl:apply-templates />}
</xsl:template>

<!-- h1.appendix (appendix) -->
<xsl:template match="div[@class='appendix']/h1">
	\newpage
	\appendix
	\section{<xsl:apply-templates />}
</xsl:template>



<!-- paragraph -->
<xsl:template match="p">
	<xsl:text>&#xa;</xsl:text>
	<xsl:apply-templates />
</xsl:template>



<!-- unordered list -->
<xsl:template match="ul">
	\begin{itemize}
		<xsl:apply-templates />
	\end{itemize}
</xsl:template>

<!-- ordered list -->
<xsl:template match="ol">
	\begin{enumerate}
		<xsl:apply-templates />
	\end{enumerate}
</xsl:template>

<!-- list item -->
<xsl:template match="li">
	\item <xsl:apply-templates />
</xsl:template>

<!-- definition list -->
<xsl:template match="dl">
	\begin{description}
		<xsl:apply-templates />
	\end{description}
</xsl:template>

<!-- definition term -->
<xsl:template match="dt">
	\item[<xsl:apply-templates />] 
</xsl:template>

<!-- definition derscription -->
<xsl:template match="dd">
	<xsl:apply-templates />
</xsl:template>



<!-- emphasized text -->
<xsl:template match="em">\emph{<xsl:apply-templates />}</xsl:template>

<!-- strong text -->
<xsl:template match="strong">\textbf{<xsl:apply-templates />}</xsl:template>

<!-- equations -->
<xsl:template match="var">$<xsl:apply-templates />$</xsl:template>

<!-- superscripts in equations -->
<xsl:template match="var/sup">^{<xsl:apply-templates />}</xsl:template>

<!-- subscripts in equations -->
<xsl:template match="var/sub">_{<xsl:apply-templates />}</xsl:template>

<!-- code -->
<xsl:template match="code">\texttt{<xsl:apply-templates />}</xsl:template>

<!-- inline quotations -->
<xsl:template match="q">\textquote{<xsl:apply-templates />}</xsl:template>

<!-- links -->
<xsl:template match="a">\href{<xsl:value-of select="@href" />}{<xsl:apply-templates />}</xsl:template>

<!-- preformatted text -->
<xsl:template match="pre">\begin{verbatim}
<xsl:apply-templates />
\end{verbatim}
</xsl:template>



<!-- table -->
<xsl:template match="table">\begin{table}
	\begin{tabular}{<xsl:for-each select="tr[1]/th">l </xsl:for-each>}
	\toprule
		<xsl:apply-templates />
	\bottomrule
	\end{tabular}
	\caption{}
\end{table}
</xsl:template>

<!-- table row -->
<xsl:template match="tr">
	<xsl:apply-templates />
</xsl:template>

<!-- table header element -->
<xsl:template match="th[position() &lt; last()]">
	<xsl:for-each select="."><xsl:apply-templates /> &amp; </xsl:for-each>
</xsl:template>

<!-- last table header element -->
<xsl:template match="th[last()]">
	<xsl:apply-templates /> \\
	\midrule
</xsl:template>

<!-- table row element -->
<xsl:template match="td[position() &lt; last()]">
	<xsl:for-each select="."><xsl:apply-templates /> &amp; </xsl:for-each>
</xsl:template>

<!-- last table row element -->
<xsl:template match="td[last()]">
	<xsl:apply-templates /> \\
</xsl:template>



<!-- footnotes -->
<xsl:key name="foo" match="/html/body/div[@id='footnotes']/ol/li" use="@id" />
<xsl:template match="sup[@class='footnote']">\footnote{<xsl:apply-templates select="key('foo', substring(a/@href, 2))/node()" />}</xsl:template>



<!-- comments -->
<xsl:template match="comment()">% <xsl:value-of select="concat(normalize-space(.), '&#xA;')" />
</xsl:template>



</xsl:stylesheet>
