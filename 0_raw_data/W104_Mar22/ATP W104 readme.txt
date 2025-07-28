PEW RESEARCH CENTER
Wave 104 American Trends Panel 
Dates: March 7 - March 13, 2022
Mode: Web
Sample: Full panel
Language: English and Spanish
N=10,441

***************************************************************************************************************************
NOTES

For a small number of respondents with high risk of identification, certain values have been randomly swapped with those of lower risk cases with similar characteristics.

The variables PARTY, PARTYLN, PARTYLN2, and PARTYSTR were asked for experimental purposes and are not included in the topline or dataset. 

The W104 dataset contains the following created variables: 
ABRT1SUM_W104
ABRT2SUM_W104
ABRTLGL6CAT_W104
ABRTLGL4CAT_W104
ATIME14WK_TOT_W104
ATIME6WK_TOT_W104
ATIME24WK_TOT_W104
ABRTN_PEN_COMBF1_W104 
ABRTN_PEN_COMBF2_W104

ABRT1SUM_W104 and ABRT2SUM_W104 summarize respondents' write-ins to ABORTIONQ1OE and ABORTIONQ2OE respectively. If a respondent wrote in multiple things that were all either pro-choice or pro-life, just their first response was coded. If the respondent mentioned multiple things that were both pro-life and pro-choice, their response was coded as "mixed."

ABRTLGL6CAT_W104, ABRTLGL4CAT_W104, ATIME14WK_TOT_W104, ATIME6WK_TOT_W104, and ATIME24WK_TOT_W104 are created variables that combine data from multiple questions on opinions towards abortion. The syntax for these created variables can be found below. 

ABRTN_PEN_COMBF1_W104 and ABRTN_PEN_COMBF2_W104 count the total amount of times a respondent said if an abortion was carried out in a situation where it was illegal, each of the following (the woman who had the abortion, the doctor or provider who performed the abortion, the person who helped find and schedule the abortion, the person who helped pay for the abortion) should face penalties. 

***************************************************************************************************************************
WEIGHTS 


WEIGHT_W104 is the weight for the sample. Data for all Pew Research Center reports are analyzed using this weight.


***************************************************************************************************************************
Releases from this survey:

March 15, 2022 "Public Expresses Mixed Views of U.S. Response to Russia's Invasion of Ukraine" 
https://www.pewresearch.org/politics/2022/03/15/public-expresses-mixed-views-of-u-s-response-to-russias-invasion-of-ukraine/

March 17, 2022. “More support than oppose Jackson’s Supreme Court nomination, with many not sure.” 
https://www.pewresearch.org/fact-tank/2022/03/17/more-support-than-oppose-jacksons-supreme-court-nomination-with-many-not-sure/

March 24, 2022 "Republicans More Likely Than Democrats to Say Partisan Control of Congress 'Really Matters'"
https://www.pewresearch.org/politics/2022/03/24/republicans-more-likely-than-democrats-to-say-partisan-control-of-congress-really-matters/

April 26, 2022. “As courts weigh affirmative action, grades and test scores seen as top factors in college admissions” 
https://www.pewresearch.org/fact-tank/2022/04/26/u-s-public-continues-to-view-grades-test-scores-as-top-factors-in-college-admissions/

May 06, 2022 "America's Abortion Quandary"
https://www.pewresearch.org/religion/2022/05/06/americas-abortion-quandary/

May 6, 2022. “Wide partisan gaps in abortion attitudes, but opinions in both parties are complicated” 
https://www.pewresearch.org/fact-tank/2022/05/06/wide-partisan-gaps-in-abortion-attitudes-but-opinions-in-both-parties-are-complicated/

June 13, 2022. “About six-in-ten Americans say abortion should be legal in all or most cases”
https://www.pewresearch.org/fact-tank/2022/06/13/about-six-in-ten-americans-say-abortion-should-be-legal-in-all-or-most-cases-2/


***************************************************************************************************************************
SYNTAX

The following syntax can be used to reporudce the variables ABRTLGL6CAT_W104, ABRTLGL4CAT_W104, ATIME14WK_TOT_W104, ATIME6WK_TOT_W104, and ATIME24WK_TOT_W104.


 compute ABRTLGL6CAT_W104=0. 
if ABRTLGL_W104=2 ABRTLGL6CAT_W104=3. 
if ABRTLGL_W104=3 ABRTLGL6CAT_W104=4. 
if ABRTLGL_W104=99 ABRTLGL6CAT_W104=99. 
if ABRTLGL1_W104=1 ABRTLGL6CAT_W104=2.
if ABRTLGL1_W104=2 ABRTLGL6CAT_W104=1.
if ABRTLGL2_W104=1 ABRTLGL6CAT_W104=5.
if ABRTLGL2_W104=2 ABRTLGL6CAT_W104=6.
if (ABRTLGL_W104=1) AND ABRTLGL1_W104=99 ABRTLGL6CAT_W104=1. 
if (ABRTLGL_W104=4) AND ABRTLGL2_W104=99 ABRTLGL6CAT_W104=6. 
val lab ABRTLGL6CAT_W104 1 "Legal in all cases, no exceptions" 
                                  2 "Legal in all case, but with some exceptions" 
                                  3 "Legal in most cases"  
                                   4 "Illegal in most cases"  
                                   5 "Illegal in all cases, but with some exceptions" 
                                   6 "Illegal in all cases, no exceptions" 
                                   99 "No answer ABRTLGL". 
var lab ABRTLGL6CAT_W104 "Detailed combination of ABRTLGL and the branching followup questions ABRTLGL1 and ABRTLGL2".


recode ABRTLGL6CAT_W104 (1=1)(2=2) (3=2) (4=3) (5=3) (6=4) (99=9) into ABRTLGL4CAT_W104. 
var lab ABRTLGL4CAT_W104 "Summary combination of ABRTLGL and the branching followup questions ABRTLGL1 and ABRTLGL2".
val lab ABRTLGL4CAT_W104 1 "Legal in all cases with no exceptions"  
                                  2 "Legal in most cases" 
                                  3 "Illegal in most cases" 
                                  4 "Illegal in all cases with no exceptions" 
                                  9 "No answer ABRTLGL".

 **the below syntax shows how to create combined detail for questions about legality of abortion at various weeks of pregnancy** 

compute ATIME14WK_TOT_W104=0.
if ABRTLGL2_W104=2 ATIME14WK_TOT_W104=100. 
if ABRTLGL2_W104=99 ATIME14WK_TOT_W104=100. 
if (ABRTTIME_W104=2 ) and ABRTLGL4CAT_W104=2 ATIME14WK_TOT_W104=2.
if (ABRTTIME_W104=2 ) and ABRTLGL4CAT_W104=3 ATIME14WK_TOT_W104=5.
if ABRTTIME_W104=99 ATIME14WK_TOT_W104=99.
if ABRTLGL1_W104=2 ATIME14WK_TOT_W104=1. 
if ABRTLGL1_W104=99 ATIME14WK_TOT_W104=1. 
if ABRTLGL_W104 =99 ATIME14WK_TOT_W104=999. 
if ATIME14WK_W104=1 ATIME14WK_TOT_W104=3.
if ATIME14WK_W104=2 ATIME14WK_TOT_W104=6.
if ATIME14WK_W104=3 ATIME14WK_TOT_W104=4.
if ATIME14WK_W104=99 ATIME14WK_TOT_W104=9.
var lab ATIME14WK_TOT_W104"Combined with responses from ATIME14WK, ABRTLGL1, ABRTLGL2, or ABRTTIME for anyone who was not asked the question". 
val lab ATIME14WK_TOT_W104 1 "Legal in all cases, no exceptions" 
                                      2 "Legal with some exceptions, but how long a woman has been pregnant should not matter"
                                      3 "Legal at 14 weeks" 
                                      4 "It depents at 14 weeks" 
                                      5 "Illegal with some exceptions, but how long a woman has been pregnant should not matter"
                                      6 "Illegal at 14 weeks"
                                      9 "Chose not to answer ATIME14WK"
                                      99 "Chose not to answer ABRTTIME"
                                     100 "Illegal in all cases, no exceptions" 
                                      999 "Chose not answer ABRTLGL". 

 compute ATIME6WK_TOT_W104=0.
if ABRTLGL2_W104=2 ATIME6WK_TOT_W104=100. 
if ABRTLGL2_W104=99 ATIME6WK_TOT_W104=100. 
if (ABRTTIME_W104=2 ) and ABRTLGL4CAT_W104=2 ATIME6WK_TOT_W104=2.
if (ABRTTIME_W104=2 ) and ABRTLGL4CAT_W104=3 ATIME6WK_TOT_W104=5.
if ABRTTIME_W104=99 ATIME6WK_TOT_W104=99.
if ABRTLGL1_W104=2 ATIME6WK_TOT_W104=1. 
if ABRTLGL1_W104=99 ATIME6WK_TOT_W104=1. 
if ABRTLGL_W104 =99 ATIME6WK_TOT_W104=999. 
if ATIME6WK_W104=1 ATIME6WK_TOT_W104=3.
if ATIME6WK_W104=2 ATIME6WK_TOT_W104=6.
if ATIME6WK_W104=3 ATIME6WK_TOT_W104=4.
if ATIME6WK_W104=99 ATIME6WK_TOT_W104=9.
if ATIME14WK_W104=1 ATIME6WK_TOT_W104=3.
var lab ATIME6WK_TOT_W104 "Combined with responses from ATIME14WK , ATIME6WK, ABRTLGL1, ABRTLGL2, or ABRTTIME for anyone who was not asked the question". 
val lab ATIME6WK_TOT_W104 1 "Legal in all cases, no exceptions" 
                                      2 "Legal in most cases, but how long a woman has been pregnant should not matter"
                                      3 "Legal at 6 weeks" 
                                      4 "It depents at 6 weeks" 
                                      5 "Illegal in most cases, but how long a woman has been pregnant should not matter"
                                      6 "Illegal at 6 weeks"
                                      9 "Chose not to answer ATIME6WK"
                                      99 "Chose not to answer ABRTTIME"
                                     100 "Illegal in all cases, no exceptions" 
                                      999 "Chose not answer ABRTLGL". 

compute ATIME24WK_TOT_W104=0.
if ABRTLGL2_W104=2 ATIME24WK_TOT_W104=100. 
if ABRTLGL2_W104=99 ATIME24WK_TOT_W104=100. 
if (ABRTTIME_W104=2 ) and ABRTLGL4CAT_W104=2 ATIME24WK_TOT_W104=2.
if (ABRTTIME_W104=2 ) and ABRTLGL4CAT_W104=3 ATIME24WK_TOT_W104=5.
if ABRTTIME_W104=99 ATIME24WK_TOT_W104=99.
if ABRTLGL1_W104=2 ATIME24WK_TOT_W104=1. 
if ABRTLGL1_W104=99 ATIME24WK_TOT_W104=1. 
if ABRTLGL_W104 =99 ATIME24WK_TOT_W104=999. 
if ATIME24WK_W104=1 ATIME24WK_TOT_W104=3.
if ATIME24WK_W104=2 ATIME24WK_TOT_W104=6.
if ATIME24WK_W104=3 ATIME24WK_TOT_W104=4.
if ATIME24WK_W104=99 ATIME24WK_TOT_W104=9.
if ATIME14WK_W104=2 ATIME24WK_TOT_W104= 6.
var lab ATIME24WK_TOT_W104 "Combined with responses from ATIME14WK , ATIME24WK, ABRTLGL1, ABRTLGL2, or ABRTTIME for anyone who was not asked the question". 
val lab ATIME24WK_TOT_W104 1 "Legal in all cases, no exceptions" 
                                      2 "Legal in most cases, but how long a woman has been pregnant should not matter"
                                      3 "Legal at 24 weeks" 
                                      4 "It depents at 24 weeks" 
                                      5 "Illegal in most cases, but how long a woman has been pregnant should not matter"
                                      6 "Illegal at 24 weeks"
                                      9 "Chose not to answer ATIME24WK_W104"
                                      99 "Chose not to answer ABRTTIME"
                                     100 "Illegal in all cases, no exceptions" 
                                      999 "Chose not answer ABRTLGL". 



The following SPSS syntax can be used to reproduce the variables ABRTN_PEN_COMBF1_W104 and ABRTN_PEN_COMBF2_W104:


COUNT ABRTN_PEN_COMBF1_W104 = ABRTN_PEN_a_W104 ABRTN_PEN_b_W104 ABRTN_PEN_c_W104 (1). 
EXECUTE. 
 
COUNT ABRTN_PEN_COMBF2_W104 = ABRTN_PEN_a_W104 ABRTN_PEN_b_W104 ABRTN_PEN_d_W104 (1). 
EXECUTE. 
VARIABLE LABELS ABRTN_PEN_COMBF1_W104 "Penalty counts. Created variable from ABRTN_PEN_a-ABRTN_PEN_c".
VARIABLE LABELS ABRTN_PEN_COMBF2_W104 "Penalty counts. Created variable from ABRTN_PEN_a-ABRTN_PEN_d".



