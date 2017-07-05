<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.0"
	xmlns:xhtml="http://www.w3.org/1999/xhtml"
	xmlns="http://www.w3.org/1999/xhtml"
	xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
	xmlns:xs="http://www.w3.org/2001/XMLSchema"
	exclude-result-prefixes="xhtml xsl xs" >

<xsl:output method="text" version="1.0" encoding="UTF-8" />

<xsl:template match="/">
<xsl:for-each select="/html/body/div/ol[@id='nav']/li/a">
<xsl:value-of select="@href" />
<xsl:text>&#xa;</xsl:text>
</xsl:for-each>
</xsl:template>

</xsl:stylesheet>
