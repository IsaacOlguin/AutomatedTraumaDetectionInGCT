###############################################################
## Description: Cleaning of transcript content depending on the 
##      Court.
## Author: Isaac Misael Olguín Nolasco
## November 2022, TUM
###############################################################

#################################################################################
### IMPORTS
import re

#################################################################################
### GLOBALS
GLB_EMPTY_STRING = ""
RE_GLB_CASE = re.IGNORECASE
RE_PARAGRAPH_WITH_B_TAG_AND_WHITE_SPACES = r'<P><B>( )*</B></P>'
RE_PARAGRAPH_WITH_B_TAG_AND_NUM_PAGES = r'<P><B>( )*Page (\d)+( )*</B></P>'
RE_PARAGRAPH_WITH_TEXT_BLANK_PAGE_INSERTED = r'(<P>( \d+|\d+)( )*Blank page inserted to ensure pagination corresponds between the French and( )*</P>)|(<P>( \d+|\d+)( )*English transcripts.( )*</P>)'
RE_PARAGRAPH_WITH_NUMBERS_AND_SPACES = r'<P>( \d+|\d+)( )*</P>'
RE_PARAGRAPH_WITH_SPACES_AT_THE_END = r'</P>( )*'
RE_PARAGRAPH_WITH_NUMBERS_AND_SPACES_AT_THE_BEGINNING = r'<P>( )*\d+( )*'
RE_SPECIFIC_TAGS = r'<HTML>|</HTML>|<center>|</center>|</FONT>|</BODY>|<FONT FACE(.*)>'
RE_PARAGRAPH_WITH_PAGE_CLOSED_SESSION = r'page (\d+) redacted – closed session'
RE_PARAGRAPH_WITH_REDACTED = r'\(redacted\)'
RE_PARAGRAPH_WITH_OPEN_OR_CLOSED_SESSION = r'\((Open|Close) session\)'
RE_PARAGRAPH_WITH_SHORT_ADJOURMENT = r'\(Short Adjournment\)'
RE_PARAGRAPH_WITH_TIMESTAMP = r'\(\d+\.\d+ (p\.m\.|a\.m\.)\)(.|( )*)|\(\d+\.\d+\)'

#################################################################################
### Cleaning of transcripts of the "International Criminal Tribunal of the 
### Former Yugoslavia"
#== @input string with the content in html format. Specifically, paragraphs 
#           between the tags "<p>" and "</p>" are received
#== @return 
def cleanParagraphsICFYtranscript(src_content):
    # Remove content that is within "b" tag with num of pages
    final_content = re.sub(RE_PARAGRAPH_WITH_B_TAG_AND_NUM_PAGES, GLB_EMPTY_STRING, src_content, flags=RE_GLB_CASE)
    # Remove spaces at the end of the paragraph
    final_content = re.sub(RE_PARAGRAPH_WITH_SPACES_AT_THE_END, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove numbers and and spaces at the beginning of the paragraph
    final_content = re.sub(RE_PARAGRAPH_WITH_NUMBERS_AND_SPACES_AT_THE_BEGINNING, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove "page \d+ redacted - closed session"
    final_content = re.sub(RE_PARAGRAPH_WITH_PAGE_CLOSED_SESSION, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove "(redacted)"
    final_content = re.sub(RE_PARAGRAPH_WITH_REDACTED, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove open or closed session
    final_content = re.sub(RE_PARAGRAPH_WITH_OPEN_OR_CLOSED_SESSION, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove short adjourment
    final_content = re.sub(RE_PARAGRAPH_WITH_SHORT_ADJOURMENT, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove timestaps within the transcript
    final_content = re.sub(RE_PARAGRAPH_WITH_TIMESTAMP, GLB_EMPTY_STRING, final_content, flags=RE_GLB_CASE)
    # Remove spaces before and after the current content
    final_content = final_content.strip()
    # Possible:
    # - (Luncheon Adjournment)

    return final_content

