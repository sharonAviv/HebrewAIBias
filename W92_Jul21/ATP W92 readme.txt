PEW RESEARCH CENTER
Wave 92 American Trends Panel 
Dates: July 8-18, 2021
Mode: Web 
Sample: Full panel
Language: English and Spanish
N=10,221

***************************************************************************************************************************
NOTES

The Wave 92 dataset contains the following computed variables. SPSSS syntax to create WHYRICHSUM_W92, GLBLZESUM_W92, and NATPROBS_TOP_W92 is included in the SYNTAX section.
R code to create TYPOLOGY_GROUP_W92 is also included in the SYNTAX section. 
 
WHYRICHSUM_W92
GLBLZESUM_W92
NATPROBS_TOP_W92
TYPOLOGY_GROUP_W92

For a small number of respondents with high risk of identification, certain values have been randomly swapped with those of lower risk cases with similar characteristics.

The 2021 Political Typology was created using weighted clustering around medoids. The items selected for inclusion in the clustering were chosen based on extensive testing to find the model that fit the data best and produced groups that were substantively meaningful.

The variable in this dataset named TYPOLOGY_GROUP_W92 represents the typology group each respondent was assigned to based on the clustering analysis.

The full model was estimated in R v4.1.1. The "WeightedCluster" package (version 1.4-1) was used for the clustering analysis and the "mice" package (version 3.13.0) was used for multiple imputation using predictive mean matching. Two other packages, used for more general data management in the original analysis, are required to run the code below: "tidyverse" and "haven".



Most of the variables used in the clustering solution are from the W92 ATP survey, but the two party "feeling thermometers" were asked in the previous wave (W91). 97% of respondents to W92 had responses from W91, but values on these questions were imputed for the remaining 3% of cases using predictive mean matching. The same procedure for multiple imputation was also used to account for the small amount of item nonresponse (ranging from 24 cases to 354 cases out of 10,221) in the W92 measures used in the model. 

Here are the variables used in the typology clustering:
DIFFPARTY_W92, GOVSIZE1_W92, GOVSIZE3_W92, USEXCEPT_W92, WOMENOBS_W92, ECONFAIR_W92, OPENIDEN_W92, ALLIES_W92, POLICY3MOD_W92, WHADVANT_W92, SUPERPWR_W92, CRIM_SENT2_W92, CANMTCHPOL_W92, PROG_RNEED_W92, PROG_RNEED2b_W92, SOCIETYTRANS_W92, PROBOFFa_W92, PROBOFFb_W92, BUSPROFIT_W92, CNTRYFAIR_W92, GOVPROTCT_W92, GOVAID_W92, RELIG_GOV_W92, RACESURV52MOD_W92, GLBLZE_W92, THERMOa_W91 (from the ATP W91 survey), and THERMOb_W91 (from the ATP W91 survey)



If you are seeking to replicate typology assignments in a different survey you will field, you can use the code below, which includes instructions for recoding and standardizing each of the 27 items used in the model to match the coding from the original clustering analysis, and then incorporates the medoid values for each group calculated in the final version of the model. Please see the accompanying doc "ATP W92 questions used in political typology model" for exact question wording for all 27 items.

NOTE: If you use this code to replicate assignments in this original W92 dataset, you will need to merge in the two feeling thermometer items from W91. Note also that there may be a small number of minor deviations between the group assignments in the cluster variable included in the public W92 dataset and those that  the code below will produce, because of the imputation used to account for missing data. If you are interested in fully replicating the typology groups from the report using this dataset, please contact us for additional syntax, which we can provide. It would also be possible to use this full code with the iterative results for the purposes of applying it to a new dataset, but this would be considerably more involved than using the code below to make assignments.


**NOTE: In past political typologies, “Bystanders” (politically disengaged Americans) were identified and excluded from the analysis out the outset. For the 2021 typology, “bystanders” were not excluded from the analysis, but the cluster modeling was performed using only data from registered voters. Although they were not used in the modeling, unregistered respondents were assigned to the group to which they were most similar and are included in the analysis.

**NOTE: Multiple custom weights were created for the report “Beyond Red vs. Blue: The Political Typology.” The weights are used for analyses combining data from multiple waves, beginning with W59 through W95 (see https://www.pewresearch.org/politics/2021/11/09/political-typology-appendix-b/ for more information). Each analysis in the report uses data from W92 (where the questions used to construct the typology were asked), and in cases where responses from another wave were needed a weight using the cases that were present in both waves was used. The weights address nonresponse from those who did not complete the additional wave and take into account panel recruitment and retirement. If you are interested in fully replicating these analyses from this report, please contact us for the additional weights and syntax, which we can provide.

***************************************************************************************************************************
WEIGHTS 

WEIGHT_W92 is the weight for the sample. Data for most Pew Research Center reports are analyzed using this weight.


***************************************************************************************************************************
Releases from this survey:

Aug. 21, 2021. “Deep Divisions in Americans’ Views of Nation’s Racial History – and How To Address It” 
https://www.pewresearch.org/politics/2021/08/12/deep-divisions-in-americans-views-of-nations-racial-history-and-how-to-address-it/

Nov. 9, 2021. “Beyond Red vs. Blue: The Political Typology” 
https://www.pewresearch.org/politics/2021/11/09/beyond-red-vs-blue-the-political-typology-2/

July 22, 2021. “Wide partisan divide on whether voting is a fundamental right or a privilege with responsibilities” 
https://www.pewresearch.org/fact-tank/2021/07/22/wide-partisan-divide-on-whether-voting-is-a-fundamental-right-or-a-privilege-with-responsibilities/

July 28, 2021. “Americans’ views about billionaires have grown somewhat more negative since 2020” 
https://pewresearch.org/fact-tank/2021/07/28/americans-views-about-billionaires-have-grown-somewhat-more-negative-since-2020/

Aug. 11, 2021. “Democrats overwhelmingly favor free college tuition, while Republicans are divided by age, education” 
https://www.pewresearch.org/fact-tank/2021/08/11/democrats-overwhelmingly-favor-free-college-tuition-while-republicans-are-divided-by-age-education/

Aug. 20, 2021. “Republicans increasingly critical of several major U.S. institutions, including big corporations and banks” 
https://www.pewresearch.org/fact-tank/2021/08/20/republicans-increasingly-critical-of-several-major-u-s-institutions-including-big-corporations-and-banks/

Aug. 26, 2021. “More Americans now say they prefer a community with big houses, even if local amenities are farther away” 
https://www.pewresearch.org/fact-tank/2021/08/26/more-americans-now-say-they-prefer-a-community-with-big-houses-even-if-local-amenities-are-farther-away/


***************************************************************************************************************************
SYNTAX

compute WHYRICHSUM_W92=0.
if WHYRICHSCALE_W92=1 or WHYRICHSCALEREV_W92=6 WHYRICHSUM_W92=1.
if WHYRICHSCALE_W92=2 or WHYRICHSCALEREV_W92=5 WHYRICHSUM_W92=2.
if WHYRICHSCALE_W92=3 or WHYRICHSCALEREV_W92=4 WHYRICHSUM_W92=3.
if WHYRICHSCALE_W92=4 or WHYRICHSCALEREV_W92=3 WHYRICHSUM_W92=4.
if WHYRICHSCALE_W92=5 or WHYRICHSCALEREV_W92=2 WHYRICHSUM_W92=5.
if WHYRICHSCALE_W92=6 or WHYRICHSCALEREV_W92=1 WHYRICHSUM_W92=6.
if WHYRICHSCALE_W92=99 or WHYRICHSCALEREV_W92=99 WHYRICHSUM_W92=99.
value labels WHYRICHSUM_W92 1'1 Rich people had more advantages in life than other people' 2'2' 3'3' 4'4' 5'5' 6'6 Rich people worked harder than other people' 99'Refused'.
VARIABLE LABELS WHYRICHSUM_W92 WHYRICHSUM_W92. Where would you place yourself on this scale? [Summary variable to de-rotate WHYRICHSCALEREV to combine with WHYRICHSCALE].

compute GLBLZESUM_W92=0.
if GLBLZE_W92=1 or GLBLZEREV_W92=2 GLBLZESUM_W92=1.
if GLBLZE_W92=2 or GLBLZEREV_W92=1 GLBLZESUM_W92=2.
if GLBLZE_W92=99 or GLBLZEREV_W92=99 GLBLZESUM_W92=99.
value labels GLBLZESUM_W92 1'Gained more than it has lost from increased trade' 2'Lost more than it has gained from increased trade' 99'Refused'.
VARIABLE LABELS GLBLZESUM_W92 GLBLZESUM_W92. All in all, would you say that the U.S. has... [Summary variable to de-rotate GLBLZEREV to combine with GLBLZE].

compute NATPROBS_TOP_W92=NATPROBS_1_W92.
do if missing(NATPROBS_1_W92).
if NATPROBS_c_w92=1 NATPROBS_TOP_W92=1.
if NATPROBS_e_w92=1 NATPROBS_TOP_W92=2.
if NATPROBS_g_w92=1 NATPROBS_TOP_W92=3.
if NATPROBS_i_w92=1 NATPROBS_TOP_W92=4.
if NATPROBS_j_w92=1 NATPROBS_TOP_W92=5.
if NATPROBS_m_w92=1 NATPROBS_TOP_W92=6.
if (NATPROBS_c_W92 > 1 and NATPROBS_e_W92 >1 and NATPROBS_g_W92 >1 and NATPROBS_i_w92 >1 and  NATPROBS_j_w92 > 1 and  NATPROBS_m_w92 > 1) NATPROBS_TOP_W92=7.
end if.
if NATPROBS_1_W92=99 NATPROBS_TOP_W92=7.
value labels NATPROBS_TOP_W92 1'Federal budget deficit' 2'Racism' 3'Illegal immigration' 4'Climate change' 5'Violent crime' 6'Economic inequality' 7'No biggest issue'.
VARIABLE LABELS NATPROBS_TOP_W92 NATPROBS_TOP_W92. Summary variable for NATPROBS biggest problem in the country today.



### R code for replication of the 2021 Political Typology
library(tidyverse)
library(haven)

### Read in full set of medoids, saved as output from PRC clustering analysis
medoids <- data.frame(group=c("Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "Group 6", "Group 7", "Group 8", "Group 9", "mean", "sd"),
                      dem_therm=c(1.257563733, 0.680334751, 0.7091962, 0.968949242, 0.103105769, -0.762737704, -0.185508722, -1.19565944, -1.339966686, 46.00030349, 34.19179159),
                      rep_therm=c(-0.695158559, -0.397911613, -1.141028979, -0.695158559, 0.19658228, 0.939699646, 0.19658228, 1.088323119, 1.088323119, 43.55186722, 33.13778692),
                      therm_diff=c(-1.952722292, -1.078246364, -1.850225179, -1.664107801, 0.093476511, 1.70243735, 0.382091002, 2.28398256, 2.428289805, NA, NA),
                      womenobs=c(-0.921271855, -0.921271855, -0.921271855, -0.921271855, 1.085445312, 1.085445312, 1.085445312, 1.085445312, 1.085445312, 1.459571527, 0.498387461),
                      society_trans=c(-0.61222219, -1.315646616, -1.315646616, -0.61222219, 0.091202236, 0.794626661, 0.091202236, 0.794626661, 0.091202236, 2.869664032, 1.414880292),
                      crim_sent2=c(-0.05157472, -1.326412007, -1.326412007, -0.05157472, -0.05157472, -0.05157472, -0.05157472, 1.223262566, 1.223262566, 2.043082524, 0.784421937),
                      relig_gov=c(-0.606761566, -0.606761566, -0.606761566, -0.606761566, -0.606761566, -0.606761566, -0.606761566, 1.648077717, -0.606761566, 1.267559524, 0.442708559),
                      policy3mod=c(-0.699095796, -0.699095796, -0.699095796, -0.699095796, -0.699095796, -0.699095796, -0.699095796, 1.071510342, 1.071510342, 1.393364929, 0.563778998),
                      canmtchpol=c(-0.725993868, 1.377408627, -0.725993868, -0.725993868, -0.725993868, -0.725993868, 1.377408627, -0.725993868, -0.725993868, 1.3451539, 0.475442112),
                      diffparty=c(-0.674814722, 0.955016151, -0.674814722, -0.674814722, 0.955016151, -0.674814722, 0.955016151, -0.674814722, -0.674814722, 1.414121575, 0.613365933),
                      proboff_a=c(-1.052025926, -1.052025926, -1.052025926, 0.652704397, 0.652704397, 0.652704397, 0.652704397, 0.652704397, 0.652704397, 2.6179941, 0.5859088),
                      proboff_b=c(-0.848145445, -0.848145445, -0.848145445, -0.848145445, -0.848145445, 0.668509195, 0.668509195, 0.668509195, 0.668509195, 1.559448819, 0.659316009),
                      econfair=c(-0.65385757, -0.65385757, -0.65385757, -0.65385757, -0.65385757, 1.5293701, 1.5293701, 1.5293701, -0.65385757, 1.299376916, 0.458007778),
                      busprofit=c(-0.789772663, -0.789772663, -0.789772663, -0.789772663, -0.789772663, 1.266174765, 1.266174765, 1.266174765, -0.789772663, 1.383618972, 0.486291085),
                      allies=c(-0.721693326, -0.721693326, -0.721693326, -0.721693326, -0.721693326, -0.721693326, -0.721693326, 1.385616549, 1.385616549, 1.341076037, 0.474094385),
                      superpwr=c(-1.33688989, -1.33688989, -1.33688989, 0.747997441, 0.747997441, 0.747997441, 0.747997441, 0.747997441, 0.747997441, 1.643356643, 0.479032493),
                      glblzesum=c(-1.162671294, -1.162671294, 0.860079905, 0.860079905, 0.860079905, -1.162671294, 0.860079905, 0.860079905, 0.860079905, 1.574268416, 0.494478387),
                      cntryfair=c(-1.078967504, -1.078967504, -1.078967504, -1.078967504, 0.926802904, 0.926802904, 0.926802904, 0.926802904, 0.926802904, 1.538852868, 0.498513033),
                      usexcept=c(-0.081148931, -1.54863518, -1.54863518, -0.081148931, -0.081148931, -0.081148931, -0.081148931, 1.386337317, -0.081148931, 2.055785124, 0.681589745),
                      govsize3=c(-0.422807561, -0.422807561, -1.715900343, -0.422807561, 0.870285221, 0.870285221, 0.870285221, 0.870285221, 0.870285221, 2.324075003, 0.774298176),
                      govprotct=c(-0.803671202, -0.803671202, -0.803671202, -0.803671202, -0.803671202, 1.244277777, 1.244277777, 1.244277777, 1.244277777, 1.391693164, 0.488152955),
                      govaid=c(-0.920219744, -0.920219744, -0.920219744, -0.920219744, -0.920219744, 1.08668633, 1.08668633, 1.08668633, 1.08668633, 1.458130525, 0.498268611),
                      openiden=c(-0.692303687, -0.692303687, -0.692303687, -0.692303687, -0.692303687, 1.444438669, -0.692303687, 1.444438669, 1.444438669, 1.318692239, 0.465992569),
                      racesurv52mod=c(-0.111883308, -1.038442452, -1.038442452, -0.111883308, -0.111883308, -0.111883308, -0.111883308, 0.814675835, -0.111883308, 2.120349465, 1.078157719),
                      prog_rneed_comb=c(-0.431153087, -1.429910131, -1.429910131, -0.431153087, -0.431153087, 0.567603956, 0.567603956, 1.566361, 0.567603956, 2.433558112, 1.005611743),
                      whadvant=c(-1.191420799, -1.191420799, -1.191420799, -1.191420799, -0.276128079, 0.63916464, 0.63916464, 1.554457359, 0.63916464, 2.298834683, 1.087923665),
                      cluster_medoid=c(13597, 10565, 10808, 12264, 8701, 4914, 6459, 3479, 1701, NA, NA),
                      cluster_ordered=c(9, 6, 7, 8, 5, 3, 4, 2, 1, NA, NA))
medoids <- column_to_rownames(medoids, var = "group")

### Read in W92 data
# Includes two items asked in W91, thermometer ratings of the Republicans and Democratic parties
dat <- read_sav("ATP W92.sav")

### Recode variables to match coding used in clustering analysis
## Includes coding for 26 of the 27 items used in the model. The 27th item, therm_diff, is calculated after transforming the data.

# Create simple recoding function
recode <- function(var, vals, labels = NULL) {
  if (class(var)[1] == "factor") lev <- levels(var)
  if (class(var)[1] != "factor") lev <- levels(factor(var))
  names(vals) <- lev
  new <- vals[as.character(var)]
  
  if (!is.null(labels)) {
    new <- factor(new, levels = sort(unique(vals)),
                  labels = labels)
  }
  return(new)
}


dat$dem_therm <- dat$THERMO_b_W91
dat$dem_therm[dat$THERMO_b_W91==998 | dat$THERMO_b_REFUSED_W91==99] <- NA

dat$rep_therm <- dat$THERMO_a_W91
dat$rep_therm[dat$THERMO_a_W91==998 | dat$THERMO_a_REFUSED_W91==99] <- NA

dat$womenobs <- recode(dat$WOMENOBS_W92, c(2, 1, NA))

dat$society_trans <- recode(dat$SOCIETY_TRANS_W92, c(1,2,3,4,5,NA))

dat$crim_sent2 <- recode(dat$CRIM_SENT2_W92, c(1,3,2,NA))

dat$relig_gov <- recode(dat$RELIG_GOV_W92, c(1,2,NA))

dat$policy3mod <- recode(dat$POLICY3MOD_W92, c(1,3,2,NA))

dat$canmtchpol <- recode(dat$CANMTCHPOL_W92, c(1,2,NA))

dat$diffparty <- recode(dat$DIFFPARTY_W92, c(1,2,3,NA))

dat$proboff_a <- recode(dat$PROBOFF_a_W92, c(3:1,NA))

dat$proboff_b <- recode(dat$PROBOFF_b_W92, c(1,2,3,NA))

dat$econfair <- recode(dat$ECONFAIR_W92, c(1,2,NA))

dat$busprofit <- recode(dat$BUSPROFIT_W92, c(1,2,NA))

dat$allies <- recode(dat$ALLIES_W92, c(1,2,NA))

dat$superpwr <-recode(dat$SUPERPWR_W92, c(2,1,NA))

dat$glblzesum <- recode(dat$GLBLZESUM_W92, c(1,2,NA))

dat$cntryfair <- recode(dat$CNTRYFAIR_W92, c(1,2,NA))

dat$usexcept <- recode(dat$USEXCEPT_W92, c(3,2,1,NA))

dat$govsize_comb <- NA
dat$govsize_comb[which(dat$GOVSIZE1_W92 == 2 & dat$GOVSIZE3_W92 == 2)] <- 1
dat$govsize_comb[which(dat$GOVSIZE1_W92 == 2 & dat$GOVSIZE3_W92 == 1)] <- 2
dat$govsize_comb[which(dat$GOVSIZE1_W92 == 1 & dat$GOVSIZE2_W92 == 2)] <- 3
dat$govsize_comb[which(dat$GOVSIZE1_W92 == 1 & dat$GOVSIZE2_W92 == 1)] <- 4
dat$govsize3 = recode(dat$govsize_comb, c(1,2,3,3))

dat$govprotct <- recode(dat$GOVPROTCT_W92, c(2,1,NA))

dat$govaid <- recode(dat$GOVAID_W92, c(2,1,NA))

dat$openiden <- recode(dat$OPENIDEN_W92, c(1,2,NA))

dat$racesurv52mod <- recode(dat$RACESURV52MOD_W92, c(4,3,2,1,NA))

dat$prog_rneed_comb <- NA
dat$prog_rneed_comb[which(dat$PROG_RNEED_W92==1 &
                            dat$PROG_RNEED2b_W92==1)] <- 1
dat$prog_rneed_comb[which(dat$PROG_RNEED_W92==1 &
                            dat$PROG_RNEED2b_W92==2)] <- 2
dat$prog_rneed_comb[which(dat$PROG_RNEED_W92==2)] <- 3
dat$prog_rneed_comb[which(dat$PROG_RNEED_W92==3)] <- 4

dat$whadvant <- recode(dat$WHADVANT_W92, c(1,2,3,4,NA))


### Transform data
zdat = array(NA, c(nrow(dat),26))
colnames(zdat) = colnames(medoids)[1:26]
for (j in 1:26) {
  nm = colnames(medoids)[j]
  if (!is.element(nm, names(dat))) next
  zdat[,j] = (dat[[nm]] - medoids['mean',nm])/medoids['sd',nm]
}

## Calculate thermometer difference
zdat[,'therm_diff'] = zdat[,'rep_therm'] - zdat[,'dem_therm']


#### Calculate typology groups based on medoids
## Get distance from each group
dist = array(0, c(nrow(dat),9))

for (j in 1:26) { ##loop through items
  for (k in 1:9) { ##loop through groups
    inds = which(!is.na(zdat[,j]))
    dist[inds,k] = dist[inds,k] + (zdat[inds,j] - medoids[k,j])^2
  }
}

### Find the minimum distance, assign each respondent to a group
dat$typo_group = apply(dist, 1, which.min)

## Re-order groups such that 1=Faith and Flag Conservatives, 2=Committed Conservatives, 3=Populist Right
## 4=Ambivalent Right, 5=Stressed Sideliners, 6=Outsider Left, 7=Democratic Mainstays, 8=Establishment Liberals, 9=Progressive Left
dat$typo_group <- recode(dat$typo_group, c(8,6,9,7,5,2,4,1,3,NA))

