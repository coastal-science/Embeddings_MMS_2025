# Collate annotations for DCLDE 2026 data
#
# Copied from JPalmerK/KillerWhaleAcousticClassification, path
# "BirdNET Models/CreateParentAnnotations.R", commit d79594c3cb5f9346a3726af35e0560566cd8e58b
# (https://github.com/JPalmerK/KillerWhaleAcousticClassification), except for
# the MALAHAT_DATA_AVAILABLE guard added below around the JASCO-Malahat section,
# fixing bug with parsing different timestamps in the orcasound data,
# and renaming path files to be up to date with the current version of DCLDE 2027 dataset stored at 
# https://console.cloud.google.com/storage/browser/noaa-passive-bioacoustic/dclde/2027

rm(list =ls())
library(lubridate)
library(dplyr)

MALAHAT_DATA_AVAILABLE <- FALSE

# Root of the source annotations/audio, and where the collated CSVs get written.
# OUTPUT_DIR defaults to its original fixed location for standalone/manual runs,
# but build_dclde_2027_annotations.py overrides it via DCLDE_OUTPUT_DIR so the
# parent CSVs land in that run's own timestamped directory.
DATA_ROOT <- '/data/DCLDE_2027'
OUTPUT_DIR <- Sys.getenv('DCLDE_OUTPUT_DIR', unset = '/home/noah/HALLO_encoder_collection/data_raw/DCLDE_2027')
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# skip running tests as not provided in original repo
# as a result all runTests calls are commented out
# source('C:/Users/kaity/Documents/GitHub/DCLDE2026/TestFx.R')

# Data to pretty
# 
# 1) Ocean Networks Canada 
# 2) Viers
# 3) DFO Cetacean Research Program (Pilkington)
# 4) DFO Whale Detection and Localization (Yerk)
# 5) SMRU
# 6) VPFA
# 7) Scripps
# 8) UAF

############################################################################
# Final output column names
set.seed(5)
colOut = c('Soundfile','Dep','LowFreqHz','HighFreqHz','FileEndSec', 'UTC',
           'FileBeginSec','ClassSpecies','KW','KW_certain','Ecotype', 'Provider',
           'AnnotationLevel', 'FilePath', 'FileOk', 'CallType')

ClassSpeciesList = c('KW', 'HW', 'AB', 'UndBio')
AnnotationLevelList = c('File', 'Detection', 'Call')
EcotypeList = c('SRKW', 'TKW', 'OKW', 'NRKW', 'SAR')
# For the class species, options are: KW, HW, AB, and UndBio
# Killer whale, humpback whale, abiotic, and unknown Bio which includes 
# Other dolphin acoustically active species ranging from fin whales, to 
# white beaked dolphins to seagulls

###############################################################################
# 1) ONC- data from globus


# Jasper, April, and Jenn have all annotated thse files.
ONC_anno = read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/onc/annotations/ONC_annotations.csv'))

ONC_anno$origEcotype = ONC_anno$Ecotype
ONC_anno$Ecotype = NA

ONC_anno$ClassSpecies = ONC_anno$Species
ONC_anno$KW_certain = NA
ONC_anno$ClassSpecies[grepl("\\|", ONC_anno$Species)]= 'UndBio'
ONC_anno$ClassSpecies[grepl("Mn", ONC_anno$Species)]= 'HW'
ONC_anno$ClassSpecies[grepl("Oo", ONC_anno$Species)]= 'KW'



# Add Ecotype 
ONC_anno$Ecotype[ONC_anno$origEcotype == 'SRKW'] ='SRKW'
ONC_anno$Ecotype[ONC_anno$origEcotype == 'TKW'] ='TKW'
ONC_anno$Ecotype[ONC_anno$origEcotype == 'OKW'] ='OKW'
ONC_anno$Ecotype[ONC_anno$origEcotype == 'BKW'] ='TKW'

# Set the ecotype to NA
ONC_anno$Ecotype[ONC_anno$ClassSpecies== 'UndBio']= NA
ONC_anno$KW = ifelse(ONC_anno$ClassSpecies=='KW',1,0)


ONC_anno$FileBeginSec= as.numeric(ONC_anno$Left.time..sec.)
ONC_anno$FileEndSec= as.numeric(ONC_anno$Right.time..sec.)
ONC_anno$HighFreqHz= as.numeric(ONC_anno$Top.freq..Hz.)
ONC_anno$LowFreqHz= as.numeric(ONC_anno$Bottom.freq..Hz.)
ONC_anno$Dep='BarkleyCanyon'

# Numerous formatting errors in the timestamp 
ONC_anno <- ONC_anno %>%
  mutate(filename = as.character(Soundfile),  # Make sure the column is treated as character
         UTC = as.POSIXct(sub(".*_(\\d{8}T\\d{6}\\.\\d{3}Z).*", "\\1", filename),
                          format = "%Y%m%dT%H%M%S.%OSZ",  tz = 'UTC'))

# Get the time of the annotation
ONC_anno$UTC = ONC_anno$UTC+ seconds(ONC_anno$FileBeginSec)

ONC_anno$Provider= 'ONC'
ONC_anno$AnnotationLevel = 'Call'
ONC_anno$dur = as.numeric(ONC_anno$FileEndSec)  - 
  as.numeric(ONC_anno$FileBeginSec)
ONC_anno$end_time = ONC_anno$UTC+ seconds(ONC_anno$dur)

# Sort and then identify overlaps
ONC_anno <- ONC_anno %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))

# There are typos in the datates and files fix them all manually
badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20130615T062356.061Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20130615T062356', 
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.061)+ONC_anno$FileBeginSec[badidx]

badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140402T032959.061Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140402T032959', 
                                  format="%Y%m%dT%H%M%S", tz="UTC")+seconds(0.061)

# Was this date hand entered? Appears to be 60 rather than 06
badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140701T605216.144Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140701T065216',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.144)+ONC_anno$FileBeginSec[badidx]

badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140702T054216.391Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140702T054216',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.391)+ONC_anno$FileBeginSec[badidx]

badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140804T1345424.361Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140804T1345424',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.361)+ONC_anno$FileBeginSec[badidx]


badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140903T231409.061Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140903T231409',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.061)+ONC_anno$FileBeginSec[badidx]


badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20141004T102246.170Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20141004T102246',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.170)+ONC_anno$FileBeginSec[badidx]

badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20141103T174455.370.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20141103T174455',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.370)+ONC_anno$FileBeginSec[badidx]


badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140104T140150.099Z-HPF.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140104T140150',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.099)+ONC_anno$FileBeginSec[badidx]



badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20140202T131456.257Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20140104T140150',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.099)+ONC_anno$FileBeginSec[badidx]



badidx = which(ONC_anno$Soundfile=='ICLISTENHF1251_20141004T102246.170Z.wav')
ONC_anno$UTC[badidx] = as.POSIXct('20141004T102246',
                                  format="%Y%m%dT%H%M%S", tz="UTC")+
  seconds(0.170)+ONC_anno$FileBeginSec[badidx]


dayFolderPath = paste0(DATA_ROOT, '/dclde_2027_killer_whales/onc/audio/barkleycanyon')
ONC_anno$FilePath = file.path(dayFolderPath,
                              ONC_anno$Soundfile)
ONC_anno$AnnotationLevel = 'call'

ONC_anno$FileOk  = file.exists(ONC_anno$FilePath)

ONC_anno$FileUTC =ONC_anno$UTC- seconds(as.numeric(ONC_anno$FileBeginSec))


ONC_anno$CallType = ONC_anno$Call.Type
ONC_anno= ONC_anno[,colOut]

rm(list= c('missingData', 'badidx', 'dayFolderPath'))

# Make sure all audio files are present for all annotations
if (all(ONC_anno$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


# runTests(ONC_anno, EcotypeList, ClassSpeciesList)



############################################################################
# DFO Cetacean Research Program
############################################################################

# No seconds in UTC
DFO_CRP1 = read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/dfo_crp/annotations/annot_H50bjRcb_SM_det.csv'))
DFO_CRP2 = read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/dfo_crp/annotations/annot_KkHK0R2F_SM_det.csv'))


DFO_CRP1$Dep='WVanIsl'
DFO_CRP2$Dep='NorthBc'


DFO_CRP = rbind(DFO_CRP1, DFO_CRP2)


DFO_CRP$UTC = as.POSIXct(DFO_CRP$utc, format="%Y-%m-%d %H:%M:%OS", tz = 'UTC')+
  seconds((as.numeric(DFO_CRP$utc_ms)/1000))

table(DFO_CRP$sound_id_species)

DFO_CRP$ClassSpecies = DFO_CRP$sound_id_species


# Clean up the abiotic counds
DFO_CRP$ClassSpecies[
  DFO_CRP$ClassSpecies %in% c('Vessel Noise', 'Unknown', '', 'Mooring Noise', 
                              'Chain?', 'ADCP', 'Anchor Noise', 'Clang',
                              'Vessel Noise?', 'Chain','No sound data',
                              "Blast/Breach", "Fishing Gear", "Breach",
                              'Rubbing',"Anthro", "Nearby Uncharted Wreck",
                              'Water Noise', "Water Noise", "Mooring" ,
                              "Nothing","Mooring?", "Naval Sonar",
                              "Rocks?", "Rock?")] = 'AB'

DFO_CRP$ClassSpecies[DFO_CRP$ClassSpecies %in% c("KW/PWSD?", "HW/KW?","KW?")]=
  'KW'

DFO_CRP$ClassSpecies[DFO_CRP$ClassSpecies %in% c("HW/GW", "HW/PWSD","HW/PWSD?",
                                                 "NPRW?HW?", 'HW?', "HW/GW?")] ='HW'

DFO_CRP$ClassSpecies[!DFO_CRP$ClassSpecies %in% ClassSpeciesList] = 'UndBio'
DFO_CRP$KW = ifelse(DFO_CRP$ClassSpecies == 'KW', 1,0)

# Set up the uncertainty
DFO_CRP$KW_certain= NA
DFO_CRP$KW_certain[DFO_CRP$KW==1] =1
DFO_CRP$KW_certain[DFO_CRP$sound_id_species %in% c("KW/PWSD?","HW/KW?",
                                                   "KW?")]=0


# Add Ecotype- note uncertain ecotypes getting their own guess
## xxx EXCLUDE UNCERTAIN ECOTYPES xxx ###
DFO_CRP$Ecotype = as.factor(DFO_CRP$kw_ecotype)
levels(DFO_CRP$Ecotype)<-c(NA, 'NRKW', NA, 'OKW', NA, 'SRKW', NA,
                            'TKW', NA, NA)
DFO_CRP$Ecotype[DFO_CRP$KW ==0]<-NA

# If the species is unsure then the ecotype is unsure
DFO_CRP$Ecotype[DFO_CRP$KW_certain==0]<- NA


DFO_CRP$Soundfile = DFO_CRP$filename
DFO_CRP$FileBeginSec = DFO_CRP$start
DFO_CRP$FileEndSec = DFO_CRP$end
DFO_CRP$LowFreqHz = DFO_CRP$freq_min
DFO_CRP$HighFreqHz = DFO_CRP$freq_max

DFO_CRP$Provider = 'DFO_CRP'
DFO_CRP$AnnotationLevel = 'Call'

DFO_CRP$dur = DFO_CRP$FileEndSec  - DFO_CRP$FileBeginSec
DFO_CRP$end_time = DFO_CRP$UTC+ seconds(DFO_CRP$dur)

# Sort and then identify overlaps
DFO_CRP <- DFO_CRP %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))


dayFolderPath_WV = paste0(DATA_ROOT, '/dclde_2027_killer_whales/dfo_crp/audio/wvanisl')
dayFolderPath_NBC = paste0(DATA_ROOT, '/dclde_2027_killer_whales/dfo_crp/audio/northbc')

DFO_CRP$FilePath = 'blarg'

DFO_CRP1$Dep='WVanIsl'
DFO_CRP2$Dep='NorthBc'


WVIidx = which(DFO_CRP$Dep == 'WVanIsl')
NBCidx = which(DFO_CRP$Dep != 'WVanIsl')

DFO_CRP$FilePath[WVIidx] =file.path(dayFolderPath_WV, DFO_CRP$Soundfile[WVIidx])
DFO_CRP$FilePath[NBCidx] = file.path(dayFolderPath_NBC, DFO_CRP$Soundfile[NBCidx])


DFO_CRP$FileOk  = file.exists(DFO_CRP$FilePath) 
# Make sure all audio files are present for all annotations
if (all(DFO_CRP$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}

# runTests(DFO_CRP, EcotypeList, ClassSpeciesList)


rm(list= c('DFO_CRP1', 'DFO_CRP2'))

DFO_CRP$CallType = DFO_CRP$call_type
DFO_CRP = DFO_CRP[, c(colOut)]


# 
# # List which files are not in annotations list
# audio.files = data.frame(
#   filename = list.files('E:/DCLDE/DFO_CRP/Audio/',
#            pattern ='.flac', recursive = TRUE, include.dirs = TRUE))
# 
# 

# audio.files$Soundfile =basename(audio.files$filename)
# audio.files$fullfile = paste0('E:/DCLDE/DFO_CRP/Audio/', audio.files$filename)
# audio.files$keep = audio.files$Soundfile %in% DFO_CRP$Soundfile 
# 
# #Move unnecesary audio to bin
# audio.files$newLoc = file.path('E:/CRP_Bin', audio.files$Soundfile)
# 
# library(filesstrings)
# for(ii in 9991:nrow(audio.files)){
#   if (audio.files$keep[ii]== FALSE){
#     move_files(audio.files$fullfile[ii], audio.files$newLoc[ii])
#     print(paste('Moved file ', audio.files$Soundfile[ii]))
#   }
# }
# 


############################################################################
# DFO Yurk
############################################################################

# Get a list of files matching the pattern 'annot_Malahat'
file_list <- list.files(path = paste0(DATA_ROOT, '/dclde_2027_killer_whales/dfo_wdlp/annotations/merged_annotations/'),
                        pattern = '*csv', full.names = TRUE)


# Read and concatenate the CSV files with filename as a separate column (if non-empty)
DFO_WDLP <- do.call(rbind, lapply(file_list, function(file) {
  data <- read.csv(file)
  if (nrow(data) > 0) {
    data$Dep <- as.factor(basename(file))  # Add filename as a new column
    return(data)
  } else {
    return(NULL)  # Return NULL for empty data frames
  }
}))

levels(DFO_WDLP$Dep)<-c('CarmanahPt', 'StrGeoN1', 'StrGeoN2','StrGeoS1',
                        'StrGeoS2','SwanChan')

# Fucking PAMGuard
DFO_WDLP = DFO_WDLP[DFO_WDLP$duration>0,]
DFO_WDLP = DFO_WDLP[!duplicated(DFO_WDLP),]

# Standardize formatting
DFO_WDLP$Soundfile = DFO_WDLP$soundfile
DFO_WDLP$LowFreqHz = DFO_WDLP$lf
DFO_WDLP$HighFreqHz = DFO_WDLP$hf
DFO_WDLP$UTC = as.POSIXct( DFO_WDLP$date_time_utc,  
                           format="%Y-%m-%d %H:%M:%OS", tz = 'UTC')

DFO_WDLP$FileBeginSec = DFO_WDLP$elapsed_time_seconds
DFO_WDLP$FileEndSec = DFO_WDLP$FileBeginSec+DFO_WDLP$duration/1000

DFO_WDLP$Ecotype = as.factor(DFO_WDLP$species)
levels(DFO_WDLP$Ecotype)<-c(NA, 'NRKW', 'SRKW', 'TKW', NA, NA)

DFO_WDLP$ClassSpecies = as.factor(DFO_WDLP$species)
levels(DFO_WDLP$ClassSpecies)<-c('HW', 'KW', 'KW', 'KW', 'UndBio', 'AB')

DFO_WDLP$KW = ifelse(DFO_WDLP$ClassSpecies== 'KW', 1,0)
DFO_WDLP$KW_certain = NA
DFO_WDLP$KW_certain[DFO_WDLP$KW==1]<-1
DFO_WDLP$Provider = 'DFO_WDA'


DFO_WDLP$AnnotationLevel = 'call'



DFO_WDLP$dur = DFO_WDLP$FileEndSec  - DFO_WDLP$FileBeginSec
DFO_WDLP$end_time = DFO_WDLP$UTC+ seconds(DFO_WDLP$dur)

# Sort and then identify overlaps
DFO_WDLP <- DFO_WDLP %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))

# Add a new column for deployment folder
DFO_WDLP$DepFolder = DFO_WDLP$Dep
levels(DFO_WDLP$DepFolder)<-c('carmanahpt',
                              'strgeon1',
                              'strgeon1',
                              'strgeos1',
                              'strgeos1',
                              'swanchan')

# Filepaths
dayFolderPath = paste0(DATA_ROOT, '/dclde_2027_killer_whales/dfo_wdlp/audio')

DFO_WDLP$FilePath = 
  file.path(dayFolderPath, 
            DFO_WDLP$DepFolder, 
            DFO_WDLP$Soundfile)


# DFO_WDLP$FilePath = 
#   file.path(dayFolderPath, DFO_WDLP$DepFolder, 
#             format(DFO_WDLP$UTC, "%Y%m%d"),
#             DFO_WDLP$Soundfile)

DFO_WDLP$FileOk  = file.exists(DFO_WDLP$FilePath) 

# Make sure all audio files are present for all annotations
if (all(DFO_WDLP$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}

# runTests(DFO_WDLP, EcotypeList, ClassSpeciesList)
DFO_WDLP$CallType = ''
DFO_WDLP = DFO_WDLP[, colOut]
rm(list=c('dayFolderPath','dayFolderPath_NBC','dayFolderPath_WV','NBCidx','WVIidx'))
   
###############################################################################
# Orca Sound Data

OrcaSound = read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/orcasound/annotations/ModifiedAnnotations.csv'))
OrcaSound$start_time_s = as.numeric(OrcaSound$start_time_s) 

OrcaSound$DateTime <- with(OrcaSound, paste(date, pst_or_master_tape_identifier))

OrcaSound$UTC <-as.POSIXct(OrcaSound$DateTime[1], 
                           tz = "America/Los_Angeles", 
                           format = "%m/%d/%Y %H:%M:%S")

# Matched by podcastId, not loop index ii: the original ii-based branches
# (e.g. `if(ii %in% c(4))`) skipped podcast_round6 and podcast_round12, so
# those groups silently reused the previous group's UTCDate via recycling
# instead of computing their own.
for(ii in 1:length(unique(OrcaSound$dataset))){

  podcastId = unique(OrcaSound$dataset)[ii]
  idx = which(OrcaSound$dataset ==podcastId)

  dateVals = OrcaSound$date[idx]
  timeVals = OrcaSound$pst_or_master_tape_identifier[idx]

  # podcast_round1's pst_or_master_tape_identifier column holds a tape ID,
  # not a clock time, so no UTC can be derived from these columns; leave NA.
  if (podcastId == 'podcast_round1') next

  if (podcastId == 'podcast_round6') {
    UTCDate = as.POSIXct(paste(dateVals, timeVals), tz = "America/Los_Angeles",
                         format = "%Y_%m_%d %H:%M:%S")+
      OrcaSound$start_time_s[idx]
  } else {
    UTCDate = as.POSIXct(paste(dateVals, timeVals),
                         tz = "America/Los_Angeles",
                         format = "%m/%d/%Y %H:%M:%S")+
      OrcaSound$start_time_s[idx]
  }

  OrcaSound$UTC[idx]<-UTCDate
}

#Now convert all times from pacific to UTC
attr(OrcaSound$UTC, "tzone") <- "UTC" 

# Add class species stuff
OrcaSound$ClassSpecies = OrcaSound$Species
OrcaSound$ClassSpecies[OrcaSound$ClassSpecies == 'Oo'] = 'KW'
OrcaSound$ClassSpecies[OrcaSound$ClassSpecies == 'Noise'] = 'AB'

OrcaSound$Provider = 'OrcaSound'
OrcaSound$KW = ifelse(OrcaSound$ClassSpecies== 'KW',1,0)
OrcaSound$KW_certain = ifelse(OrcaSound$ClassSpecies== 'KW',1,NA)
OrcaSound$Dep = OrcaSound$location
OrcaSound$Soundfile = OrcaSound$wav_filename
OrcaSound$LowFreqHz =NaN
OrcaSound$HighFreqHz =NaN

OrcaSound$FileBeginSec = OrcaSound$start_time_s
OrcaSound$FileEndSec = OrcaSound$start_time_s+OrcaSound$duration_s


OrcaSound$AnnotationLevel = ifelse(OrcaSound$ClassSpecies == 'KW',
                                   'Detection', 'File')
levels(OrcaSound$location)


# Filepaths
dayFolderPath = paste0(DATA_ROOT, '/dclde_2027_killer_whales/orcasound/audio')
OrcaSound$FilePath = 
  file.path(dayFolderPath,OrcaSound$Dep, OrcaSound$Soundfile)

OrcaSound$FileOk  = file.exists(OrcaSound$FilePath) 

# Make sure all audio files are present for all annotations
if (all(OrcaSound$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


# 
# OrcaSound[OrcaSound$AnnotationLevel=='File', 
#           c('FileBeginSec', 'FileEndSec',
#             'UTC', 'LowFreqHz', 'HighFreqHz')]= NA

OrcaSound$CallType =''

OrcaSound = OrcaSound[,c(colOut)]

# runTests(OrcaSound, EcotypeList, ClassSpeciesList)


###########################################################################
# SIMRES
##########################################################################


# multiple selection tables
file_list <- list.files(path = paste0(DATA_ROOT, '/dclde_2027_killer_whales/simres/annotations/'),
                        pattern = '*txt', full.names = TRUE)


# Read and concatenate the CSV files with filename as a separate column (if non-empty)
SIMRES <- do.call(rbind, lapply(file_list, function(file) {
  data <- read.table(file, header = TRUE, sep = '\t')
  if (nrow(data) > 0) {
    data$Dep <- basename(file)  # Add filename as a new column
    return(data)
  } else {
    return(NULL)  # Return NULL for empty data frames
  }
}))

# Clean out blank rows
SIMRES = SIMRES[!is.na(SIMRES$Selection),]

SIMRES$ClassSpecies = as.factor(SIMRES$Sound.ID.Species)
levels(SIMRES$ClassSpecies)<-c('AB', 'HW','KW',  'KW', 'KW')


#check SIMRES files in UTC
SIMRES$UTC = as.POSIXct(sub(".*_(\\d{8}T\\d{6}\\.\\d{3}Z)_.*", "\\1", 
                            SIMRES$Begin.File),  
                        format = "%Y%m%dT%H%M%S.%OSZ",
                        tz = 'UTC')+seconds(SIMRES$File.Offset..s.)

SIMRES$Ecotype = as.factor(SIMRES$KW.Ecotype)
levels(SIMRES$Ecotype)<- c(NA, NA, 'SRKW', 'SRKW')

SIMRES$KW =ifelse(SIMRES$ClassSpecies=='KW', 1, 0)

SIMRES$KW_certain = NA
SIMRES$KW_certain[SIMRES$KW ==1] =1
SIMRES$KW_certain[SIMRES$Confidence %in% c('low')]=0


SIMRES$Soundfile = SIMRES$Begin.File
SIMRES$Dep = 'Tekteksen'
SIMRES$LowFreqHz = SIMRES$Low.Freq..Hz.
SIMRES$HighFreqHz = SIMRES$Low.Freq..Hz.
SIMRES$FileBeginSec = SIMRES$File.Offset..s.
SIMRES$FileEndSec = SIMRES$File.Offset..s.+SIMRES$Delta.Time..s.
SIMRES$Provider = 'SIMRES'
SIMRES$AnnotationLevel = 'Call'


# Filepaths- wackadoodle for SIMRES
dayFolderPath = paste0(DATA_ROOT, '/dclde_2027_killer_whales/simres/audio/')

# 1. List all files in the target directory
allAudio <- list.files(path = dayFolderPath, pattern = "\\.flac$", 
                       full.names = TRUE, recursive = TRUE)

# 2. Check if each file exists and get the full path
SIMRES$FilePath <- sapply(SIMRES$Soundfile, function(filename) {
  # Find the full path of the file (if it exists)
  full_path <- allAudio[grep(paste0("/", filename, "$"), allAudio)]
  if (length(full_path) > 0) return(full_path[1]) else return(NA)
})



SIMRES <- SIMRES %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(SIMRES$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


# runTests(SIMRES, EcotypeList, ClassSpeciesList)

SIMRES$CallType = SIMRES$Call.Type

SIMRES= SIMRES[, colOut]
rm(list= c('file_list', 'dayFolderPath'))

############################################################################
# VFPA - JASCO Strait of Georgia (Roberts Bank in Globus)
############################################################################

# Strait fo Georgia
VPFA_SoG<- read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/annotations/annot_RB_man_det.csv'))

VPFA_SoG <- VPFA_SoG %>%
  mutate(
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}.\\d{3}Z).*", "\\1", filename), 
                     format = "%Y%m%dT%H%M%S", tz= 'UTC')
  )


VPFA_SoG$ClassSpecies=VPFA_SoG$sound_id_species
VPFA_SoG$ClassSpecies[VPFA_SoG$sound_id_species == 'KW?'] ='KW'

VPFA_SoG$ClassSpecies[VPFA_SoG$sound_id_species %in% c('KW?', "HW/KW?",
                                                       'KW/PWSD?')]<-'KW'
VPFA_SoG$ClassSpecies[VPFA_SoG$sound_id_species %in% c( "HW?")]<-'HW'

VPFA_SoG$ClassSpecies[VPFA_SoG$ClassSpecies %in% 
                        c("Vessel Noise", "Vessel Noise?",  "Noise", 
                          "Sonar","UN")]= 'AB'

VPFA_SoG$ClassSpecies[VPFA_SoG$ClassSpecies %in% 
                        c("PWSD","FS")]= 'UndBio'



VPFA_SoG$KW = ifelse(VPFA_SoG$ClassSpecies == 'KW',1,0)
VPFA_SoG$KW_certain= NA
VPFA_SoG$KW_certain[VPFA_SoG$KW==1] =1
VPFA_SoG$KW_certain[VPFA_SoG$KW==1 & 
                      grepl("\\?", VPFA_SoG$sound_id_species)]<-0
VPFA_SoG$KW_certain[VPFA_SoG$sound_id_species %in% c('KW?', "HW/KW?",
                                                     'KW/PWSD?')]<-0



VPFA_SoG$UTC = VPFA_SoG$UTC+ seconds(as.numeric(VPFA_SoG$start))

# Add Ecotype 
# XXX EXCLUDE UNCERTAIN ECOTYPES XXX
VPFA_SoG$Ecotype = NA
VPFA_SoG$Ecotype[VPFA_SoG$kw_ecotype == 'SRKW'] ='SRKW'
VPFA_SoG$Ecotype[VPFA_SoG$kw_ecotype == 'SRKW?'] = NA
VPFA_SoG$Ecotype[VPFA_SoG$kw_ecotype == 'KWT?'] = NA
VPFA_SoG$Ecotype[VPFA_SoG$kw_ecotype == 'TKW?'] =NA


colnames(VPFA_SoG)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                           'FileEndSec', 'Soundfile')


VPFA_SoG$AnnotationLevel = 'Call'
VPFA_SoG$dur = VPFA_SoG$FileEndSec-VPFA_SoG$FileBeginSec
VPFA_SoG$end_time = VPFA_SoG$UTC+ seconds(VPFA_SoG$dur)

VPFA_SoG$Dep='StraitofGeorgia'
VPFA_SoG$Provider = 'JASCO_VFPA'

# Sort and then identify overlaps
VPFA_SoG <- VPFA_SoG %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))


# List which files are not in annotations list
audio.files = data.frame(
  filename = list.files(paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/audio/straitofgeorgia_globus-robertsbank/'),
                        pattern ='.wav', recursive = TRUE, include.dirs = TRUE))
audio.files$Soundfile =basename(audio.files$filename)

# Create subset of high pass filtered files
hpf_df <- audio.files[grep("hpf", audio.files$filename, ignore.case = TRUE), ]

# 
# # Print the subset
# print(hpf_filenames)




# Day folder
dayFolderPath = paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/audio/straitofgeorgia_globus-robertsbank/')
VPFA_SoG$FilePath = file.path(dayFolderPath,
                              VPFA_SoG$Soundfile)

# VPFA_SoG$FilePath = file.path(dayFolderPath,
#                               format(VPFA_SoG$UTC-seconds(VPFA_SoG$FileBeginSec), 
#                                      "%Y%m%d"),
#                               VPFA_SoG$Soundfile)

VPFA_SoG <- VPFA_SoG %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(VPFA_SoG$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


VPFA_SoG$CallType = VPFA_SoG$call_type
VPFA_SoG =VPFA_SoG[,colOut]

# runTests(VPFA_SoG, EcotypeList, ClassSpeciesList)

############################################################################
# VFPA - JASCO Boundary Pass
############################################################################


# Boundary Pass
VPFA_BoundaryPass<- read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/annotations/annot_BP_man_det.csv'))
VPFA_BoundaryPass <- VPFA_BoundaryPass %>%
  mutate(
    KW = as.numeric(grepl("KW", sound_id_species)),
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}Z).*", "\\1", filename), 
                     format = "%Y%m%dT%H%M%S", tz= 'UTC')
  )



# Remove 'duplicate' and 'repeat' annotations
VPFA_BoundaryPass= VPFA_BoundaryPass[
  !VPFA_BoundaryPass$sound_id_species %in% c('Repeat', 'Duplicate'),]


VPFA_BoundaryPass$ClassSpecies<- VPFA_BoundaryPass$sound_id_species


VPFA_BoundaryPass$ClassSpecies[VPFA_BoundaryPass$sound_id_species %in%
                                 c('KW?', "HW/KW?",  "KW/HW?","KW/PWSD?")] ='KW'

VPFA_BoundaryPass$ClassSpecies[VPFA_BoundaryPass$sound_id_species %in% 
                                 c( "HW?")]= 'HW'

VPFA_BoundaryPass$ClassSpecies[VPFA_BoundaryPass$sound_id_species %in% 
                                 c("Vessel Noise", "Vessel Noise?",  "Noise", 
                                   "Sonar","UN", "NN", "BACKGROUND", 'UNK')]= 'AB'

VPFA_BoundaryPass$ClassSpecies[VPFA_BoundaryPass$sound_id_species %in% 
                                 c("PWSD","FS",  "PWSD?")]= 'UndBio'
VPFA_BoundaryPass$KW = ifelse(VPFA_BoundaryPass$ClassSpecies== 'KW', 1,0)
VPFA_BoundaryPass$KW_certain= NA
VPFA_BoundaryPass$KW_certain[VPFA_BoundaryPass$KW==1] =1
UncertainKWidx = which(VPFA_BoundaryPass$KW==1 & 
                         grepl("\\?", VPFA_BoundaryPass$sound_id_species))
VPFA_BoundaryPass$KW_certain[UncertainKWidx]=0

# Get time in UTC
VPFA_BoundaryPass$UTC = VPFA_BoundaryPass$UTC+ 
  seconds(as.numeric(VPFA_BoundaryPass$start))

# Add Ecotype XX EXCLUDE UNCERTAIN ECOTYPES XX
VPFA_BoundaryPass$Ecotype = NA
VPFA_BoundaryPass$Ecotype[VPFA_BoundaryPass$kw_ecotype == 'SRKW'] ='SRKW'
VPFA_BoundaryPass$Ecotype[VPFA_BoundaryPass$kw_ecotype == 'SRKW?'] =NA
VPFA_BoundaryPass$Ecotype[VPFA_BoundaryPass$kw_ecotype == 'KWT?'] =NA
VPFA_BoundaryPass$Ecotype[VPFA_BoundaryPass$kw_ecotype == 'TKW?'] =NA
VPFA_BoundaryPass$Ecotype[VPFA_BoundaryPass$kw_ecotype == 'TKW'] ='TKW'



colnames(VPFA_BoundaryPass)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                    'FileEndSec', 'Soundfile')



VPFA_BoundaryPass$AnnotationLevel = 'Call'
VPFA_BoundaryPass$dur = VPFA_BoundaryPass$FileEndSec-VPFA_BoundaryPass$FileBeginSec
VPFA_BoundaryPass$end_time = VPFA_BoundaryPass$UTC+ seconds(VPFA_BoundaryPass$dur)
VPFA_BoundaryPass$Dep='BoundaryPass'
VPFA_BoundaryPass$Provider = 'JASCO_VFPA'



# List which files are not in annotations list
audio.files = data.frame(
  filename = list.files(paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/audio/boundarypass/'),
                        pattern ='.wav', recursive = TRUE, include.dirs = TRUE))
audio.files$Soundfile =basename(audio.files$filename)

# Make sure all audio files are present for all annotations
if (all(VPFA_BoundaryPass$Soundfile %in% audio.files$Soundfile)){
  print('All data present for annotations')
  VPFA_BoundaryPass$FileOk= TRUE
}else{
  print('Missing data')
  VPFA_BoundaryPass$Soundfile[which(!VPFA_BoundaryPass$Soundfile %in% audio.files$Soundfile)]
}




# Day folder
dayFolderPath = paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/audio/boundarypass')
VPFA_BoundaryPass$FilePath = file.path(dayFolderPath,
  VPFA_BoundaryPass$Soundfile)

# VPFA_BoundaryPass$FilePath = file.path(dayFolderPath, format(
#   VPFA_BoundaryPass$UTC-seconds(VPFA_BoundaryPass$FileBeginSec),"%Y%m%d"),
#                               VPFA_BoundaryPass$Soundfile)

VPFA_BoundaryPass <- VPFA_BoundaryPass %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(VPFA_BoundaryPass$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}

VPFA_BoundaryPass$CallType = VPFA_BoundaryPass$call_type

VPFA_BoundaryPass = VPFA_BoundaryPass[,colOut]
# runTests(VPFA_BoundaryPass, EcotypeList, ClassSpeciesList)

############################################################################
# VFPA - JASCO Haro Strait North
############################################################################


# Haro Strait North
VPFA_HaroNB<- read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/annotations/annot_VFPA-HaroStrait-NB_SM_coarse.csv'))
VPFA_HaroNB <- VPFA_HaroNB %>%
  mutate(
    KW = as.numeric(grepl("KW", sound_id_species)),
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}Z).*", "\\1", filename), 
                     format = "%Y%m%dT%H%M%S", tz= 'UTC')
  )

VPFA_HaroNB$ClassSpecies<- VPFA_HaroNB$sound_id_species

VPFA_HaroNB$ClassSpecies[VPFA_HaroNB$sound_id_species %in%
                                 c('KW?', "HW/KW?",  "KW/HW?","KW/PWSD?")] ='KW'

VPFA_HaroNB$ClassSpecies[VPFA_HaroNB$sound_id_species %in% 
                                 c( "HW?")]= 'HW'

VPFA_HaroNB$ClassSpecies[VPFA_HaroNB$sound_id_species %in% 
                           c("Vessel Noise", "Vessel Noise?",  "Noise", 
                             "Sonar","UN", "BELL","VESSEL", "UNK")]= 'AB'

VPFA_HaroNB$ClassSpecies[VPFA_HaroNB$sound_id_species %in% 
                                 c("PWSD","FS",  "PWSD?")]= 'UndBio'

VPFA_HaroNB$KW_certain= NA
VPFA_HaroNB$KW_certain[VPFA_HaroNB$KW==1] =1

UncertainKWidx = which(VPFA_HaroNB$KW==1 & 
                         grepl("\\?", VPFA_HaroNB$sound_id_species))
VPFA_HaroNB$KW_certain[UncertainKWidx]=0



# Get time in UTC
VPFA_HaroNB$UTC = VPFA_HaroNB$UTC+ 
  seconds(as.numeric(VPFA_HaroNB$start))

# Add Ecotype XXX EXCLUDE UNCERTAIN ECOTYPES
VPFA_HaroNB$Ecotype = NA
VPFA_HaroNB$Ecotype[VPFA_HaroNB$kw_ecotype == 'SRKW'] ='SRKW'
VPFA_HaroNB$Ecotype[VPFA_HaroNB$kw_ecotype == 'SRKW?'] =NA
VPFA_HaroNB$Ecotype[VPFA_HaroNB$kw_ecotype == 'KWT?'] = NA
VPFA_HaroNB$Ecotype[VPFA_HaroNB$kw_ecotype == 'TKW?'] =NA



colnames(VPFA_HaroNB)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                             'FileEndSec', 'Soundfile')

VPFA_HaroNB$AnnotationLevel = 'Call'
VPFA_HaroNB$dur = VPFA_HaroNB$FileEndSec-VPFA_HaroNB$FileBeginSec
VPFA_HaroNB$end_time = VPFA_HaroNB$UTC+ seconds(VPFA_HaroNB$dur)
VPFA_HaroNB$Dep='HaroStraitNorth'
VPFA_HaroNB$Provider = 'JASCO_VFPA'

# List which files are not in annotations list
audio.files = data.frame(
  filename = list.files(paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/audio/vfpa-harostrait-nb/'),
                        pattern ='.wav', recursive = TRUE, include.dirs = TRUE))
audio.files$Soundfile =basename(audio.files$filename)

# Make sure all audio files are present for all annotations
if (all(VPFA_HaroNB$Soundfile %in% audio.files$Soundfile)){
  print('All data present for annotations')
  VPFA_HaroNB$FileOk=TRUE
}else{
  print('Missing data')
  VPFA_HaroNB$Soundfile[which(!VPFA_HaroNB$Soundfile %in% audio.files$Soundfile)]
}



# Day folder
dayFolderPath = paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/audio/vfpa-harostrait-nb/')
VPFA_HaroNB$FilePath = file.path(dayFolderPath,
  VPFA_HaroNB$Soundfile)

# VPFA_HaroNB$FilePath = file.path(dayFolderPath, format(
#   VPFA_HaroNB$UTC-seconds(VPFA_HaroNB$FileBeginSec),"%Y%m%d"),
#   VPFA_HaroNB$Soundfile)

VPFA_HaroNB <- VPFA_HaroNB %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(VPFA_HaroNB$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


VPFA_HaroNB$CallType = VPFA_HaroNB$signal_type
# runTests(VPFA_HaroNB, EcotypeList, ClassSpeciesList)

VPFA_HaroNB= VPFA_HaroNB[,colOut]


############################################################################
# VFPA - JASCO Haro Strait South
############################################################################

# Haro Strait South
VPFA_HaroSB<- read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/annotations/annot_VFPA-HaroStrait-SB_SM_coarse.csv'))
VPFA_HaroSB <- VPFA_HaroSB %>%
  mutate(
    KW = as.numeric(grepl("KW", sound_id_species)),
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}Z).*", "\\1", filename), 
                     format = "%Y%m%dT%H%M%S", tz= 'UTC')
  )



VPFA_HaroSB$ClassSpecies<- VPFA_HaroSB$sound_id_species


VPFA_HaroSB$ClassSpecies[VPFA_HaroSB$sound_id_species %in%
                           c('KW?', "HW/KW?",  "KW/HW?","KW/PWSD?")] ='KW'

VPFA_HaroSB$ClassSpecies[VPFA_HaroSB$sound_id_species %in% 
                           c( "HW?")]= 'HW'

VPFA_HaroSB$ClassSpecies[VPFA_HaroSB$sound_id_species %in% 
                           c("Vessel Noise", "Vessel Noise?",  "Noise", 
                             "Sonar","UN", "BELL","VESSEL", "UNK")]= 'AB'

VPFA_HaroSB$ClassSpecies[VPFA_HaroSB$sound_id_species %in% 
                           c("PWSD","FS",  "PWSD?")]= 'UndBio'

VPFA_HaroSB$KW_certain= NA
VPFA_HaroSB$KW_certain[VPFA_HaroSB$KW==1] =1

UncertainKWidx = which(VPFA_HaroSB$KW==1 & 
                         grepl("\\?", VPFA_HaroSB$sound_id_species))
VPFA_HaroSB$KW_certain[UncertainKWidx]=0



# Get time in UTC
VPFA_HaroSB$UTC = VPFA_HaroSB$UTC+ 
  seconds(as.numeric(VPFA_HaroSB$start))

# Add Ecotype XXX ECLUCED UNCERTAN ECOTYPES
VPFA_HaroSB$Ecotype = NA
VPFA_HaroSB$Ecotype[VPFA_HaroSB$kw_ecotype == 'SRKW'] ='SRKW'
VPFA_HaroSB$Ecotype[VPFA_HaroSB$kw_ecotype == 'SRKW?'] =NA
VPFA_HaroSB$Ecotype[VPFA_HaroSB$kw_ecotype == 'KWT?'] = NA
VPFA_HaroSB$Ecotype[VPFA_HaroSB$kw_ecotype == 'TKW?'] = NA


colnames(VPFA_HaroSB)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                        'FileEndSec', 'Soundfile')


VPFA_HaroSB$AnnotationLevel = 'Call'
VPFA_HaroSB$dur = VPFA_HaroSB$FileEndSec-VPFA_HaroSB$FileBeginSec
VPFA_HaroSB$end_time = VPFA_HaroSB$UTC+ seconds(VPFA_HaroSB$dur)
VPFA_HaroSB$Dep='HaroStraitSouth'
VPFA_HaroSB$Provider = 'JASCO_VFPA'



# List which files are not in annotations list
audio.files = data.frame(
  filename = list.files(paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/audio/vfpa-harostrait-sb/'),
                        pattern ='.wav', recursive = TRUE, include.dirs = TRUE))
audio.files$Soundfile =basename(audio.files$filename)

# Make sure all audio files are present for all annotations
if (all(VPFA_HaroSB$Soundfile %in% audio.files$Soundfile)){
  print('All data present for annotations')
  VPFA_HaroSB$FileOk = TRUE
}else{
  print('Missing data')
  VPFA_HaroSB$Soundfile[which(!VPFA_HaroSB$Soundfile %in% audio.files$Soundfile)]
}

# Day folder
dayFolderPath = paste0(DATA_ROOT, '/dclde_2027_killer_whales/vfpa/audio/vfpa-harostrait-sb/')
VPFA_HaroSB$FilePath = file.path(dayFolderPath, 
  VPFA_HaroSB$Soundfile)

# VPFA_HaroSB$FilePath = file.path(dayFolderPath, format(
#   VPFA_HaroSB$UTC-seconds(VPFA_HaroSB$FileBeginSec),"%Y%m%d"),
#   VPFA_HaroSB$Soundfile)

VPFA_HaroSB <- VPFA_HaroSB %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(VPFA_HaroSB$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


VPFA_HaroSB$CallType = VPFA_HaroSB$signal_type

VPFA_HaroSB= VPFA_HaroSB[, colOut]
# runTests(VPFA_HaroSB, EcotypeList, ClassSpeciesList)

###########################################################################
# SCRIPPS
############################################################################


# Get a list of files matching the pattern 'annot_Malahat'
file_list <- list.files(path = paste0(DATA_ROOT, '/dclde_2027_killer_whales/scripps/annotations'),
                        pattern = '*txt', full.names = TRUE)


# Read and concatenate the CSV files with filename as a separate column (if non-empty)
scripps <- do.call(rbind, lapply(file_list, function(file) {
  data <- read.table(file, header = TRUE, sep = '\t')
  if (nrow(data) > 0) {
    data$Dep <- as.factor(basename(file))  # Add filename as a new column
    return(data)
  } else {
    return(NULL)  # Return NULL for empty data frames
  }
}))

scripps$Dep<- as.factor(scripps$Dep)

# Cape Elizabeth and Quinault Canyon
levels(scripps$Dep)<-c("ce", "qc")

# Set the initial ecotypes then match the format
scripps$Ecotype <- as.factor(gsub(".*_(.*)\\.wav", "\\1", scripps$Begin.File))
levels(scripps$Ecotype)<-c('TKW', 'TKW', 'TKW', 'TKW', 'OKW', 'SRKW',
                           'SRKW','TKW', 'TKW')
scripps$Ecotype[scripps$ClassSpecies != ""]<- NaN

scripps$ClassSpecies= as.factor(scripps$ClassSpecies)
levels(scripps$ClassSpecies)<-c('KW', 'AB', 'AB', 'HW', 'KW')

scripps$KW = ifelse(scripps$ClassSpecies=='KW',1,0)
scripps$KW_certain = ifelse(scripps$ClassSpecies=='KW',1,NA)

scripps$Soundfile = scripps$Begin.File
scripps$LowFreqHz = scripps$Low.Freq..Hz.
scripps$HighFreqHz = scripps$Low.Freq..Hz.
scripps$FileBeginSec = scripps$Beg.File.Samp..samples./200000
scripps$FileEndSec = scripps$FileBeginSec+(scripps$End.Time..s.- scripps$Begin.Time..s)
scripps$Provider = 'SIO'
scripps$AnnotationLevel = 'Call'
scripps$FilePath = paste0(paste0(DATA_ROOT, '/dclde_2027_killer_whales/scripps/audio/'), scripps$Dep, '/',
                          scripps$Soundfile)


#check SIMRES files in UTC
scripps$UTC = as.POSIXct(sub(".*_(\\d{6}_\\d{6})_.*", "\\1",
                             scripps$Begin.File),  
                        format = "%y%m%d_%H%M%S",
                        tz = 'UTC')

scripps$FileOk = file.exists(scripps$FilePath)

# Make sure all audio files are present for all annotations
if (all(scripps$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}




scripps$CallType = ""

scripps = scripps[, colOut]
# runTests(scripps, EcotypeList, ClassSpeciesList)

#############################################################################
# SMRU Consulting
#############################################################################


SMRU_SRKW<- read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/smru/annotations/annot_LimeKiln-Encounters_man_det.csv'))
SMRU_HW<- read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/smru/annotations/annot_LimeKiln-Humpback_man_det.csv'))

SMRU <- rbind(SMRU_HW, SMRU_SRKW)

SMRU <- SMRU %>%
  mutate(
    UTC = as.POSIXct(sub(".*_(\\d{8}_\\d{6}_\\d{3})\\..*", "\\1", filename), 
               format = "%Y%m%d_%H%M%S_%OS", tz = "UTC")
  )

# Some annotations witout sound species ID, remove
SMRU = SMRU[SMRU$sound_id_species != "",]

SMRU$ClassSpecies=SMRU$sound_id_species
SMRU$ClassSpecies[SMRU$sound_id_species %in% 
                    c('KW?', "KW", "HW/KW?", "KW/PWSD","KW/PWSD?",
                      "PWSD/KW?")]<-'KW'

SMRU$ClassSpecies[SMRU$sound_id_species %in% c( "HW?", "HW", "HB", "HB?")]<-'HW'

SMRU$ClassSpecies[SMRU$ClassSpecies %in% 
                        c("Vessel Noise", "Vessel Noise?",  "Noise", 
                          "Sonar","UN", "Unknown", "Mooring", "surface noise",
                          "Vessel noise", "Unknown?", "Mooring noise", 
                          "Wave noise", "Vessel  Noise", "Surf noise",
                          "Waves", "Mooring ", "Mooring?")]= 'AB'

SMRU$ClassSpecies[SMRU$ClassSpecies %in% 
                        c("PWSD","FS", "Seal?", "Fish", "Fish?", "FISH",
                          "Snapping shrimp/urchin", 
                          "Snapping shrimp or urchin" )]= 'UndBio'


# Set the KW class and certainty
SMRU$KW = ifelse(SMRU$ClassSpecies == 'KW',1,0)
SMRU$KW_certain= NA
SMRU$KW_certain[SMRU$KW==1] =1
SMRU$KW_certain[SMRU$KW==1 & grepl("\\?", SMRU$sound_id_species)]<-0


SMRU$UTC = SMRU$UTC+ seconds(as.numeric(SMRU$start))

# Add Ecotype XXX EXCLUDE UNCERTAIN ECOTYPES
SMRU$Ecotype = NA
SMRU$Ecotype[SMRU$kw_ecotype == 'SRKW'] ='SRKW'
SMRU$Ecotype[SMRU$kw_ecotype == 'SRKW?'] =NA
SMRU$Ecotype[SMRU$kw_ecotype == 'KWT?'] =NA
SMRU$Ecotype[SMRU$kw_ecotype == 'TKW?'] =NA


colnames(SMRU)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                    'FileEndSec', 'Soundfile')


SMRU$AnnotationLevel = 'Call'

SMRU$dur = SMRU$FileEndSec-SMRU$FileBeginSec
SMRU$end_time = SMRU$UTC+ seconds(SMRU$dur)

SMRU$Dep='LimeKiln'
SMRU$Provider = 'SMRUConsulting'

# Sort and then identify overlaps
SMRU <- SMRU %>%
  arrange(Dep,UTC) %>%
  mutate(overlap = lead(UTC) <= lag(end_time, default = first(end_time)))


# List which files are not in annotations list
audio.files = data.frame(
  filename = list.files(paste0(DATA_ROOT, '/dclde_2027_killer_whales/smru/audio/lime-kiln/'),
                        pattern ='.wav', recursive = TRUE, include.dirs = TRUE))
audio.files$Soundfile =basename(audio.files$filename)



# Day folder
dayFolderPath = paste0(DATA_ROOT, '/dclde_2027_killer_whales/smru/audio/lime-kiln/')
SMRU$FilePath = file.path(dayFolderPath,
                          SMRU$Soundfile)

SMRU <- SMRU %>%
  mutate(FileOk = file.exists(FilePath))

# Make sure all audio files are present for all annotations
if (all(SMRU$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}


SMRU$CallType = SMRU$call_type

SMRU =SMRU[,colOut]

# runTests(SMRU, EcotypeList, ClassSpeciesList)


##############################################################################
#JASCO- Malahat
###############################################################################
if (MALAHAT_DATA_AVAILABLE) {

# Get a list of files matching the pattern 'annot_Malahat'
file_list <- list.files(path = 'E:/DCLDE/Malahat_JASCO/Annotations/',
                        pattern = 'annot_Malahat', full.names = TRUE)


# Read and concatenate the CSV files with filename as a separate column (if non-empty)
JASCO_malahat <- do.call(rbind, lapply(file_list, function(file) {
  data <- read.csv(file)
  if (nrow(data) > 0) {
    data$Dep <- as.factor(basename(file))  # Add filename as a new column
    return(data)
  } else {
    return(NULL)  # Return NULL for empty data frames
  }
}))


JASCO_malahat <- JASCO_malahat %>%
  mutate(
    UTC = as.POSIXct(sub(".*(\\d{8}T\\d{6}Z).*", "\\1", filename),
                     format = "%Y%m%dT%H%M%S", tz= 'UTC'))

JASCO_malahat$UTC = JASCO_malahat$UTC+
  seconds(as.numeric(JASCO_malahat$start))

JASCO_malahat$ClassSpecies = as.factor(JASCO_malahat$sound_id_species)
levels(JASCO_malahat$ClassSpecies)<-c('AB', 'HW','KW','HW',  'KW', 'KW', 
                                      'UndBio', 'UndBio','AB', 'AB', 'AB')

JASCO_malahat$Ecotype = as.factor(JASCO_malahat$kw_ecotype)
levels(JASCO_malahat$Ecotype)<- c(NA,'SRKW', NA, 'TKW', NA)

JASCO_malahat$KW =ifelse(JASCO_malahat$ClassSpecies %in% c('KW', 'KW?'), 1, 0)

JASCO_malahat$KW_certain = NA
JASCO_malahat$KW_certain[JASCO_malahat$KW ==1] =1
JASCO_malahat$KW_certain[JASCO_malahat$sound_id_species %in% c("HW/KW?",
                                                               "KW?")]=0

colnames(JASCO_malahat)[c(5,6,3,4,1)]<-c('LowFreqHz','HighFreqHz','FileBeginSec',
                                         'FileEndSec', 'Soundfile')

JASCO_malahat$AnnotationLevel = 'Call'

JASCO_malahat$dur = JASCO_malahat$FileEndSec-JASCO_malahat$FileBeginSec

JASCO_malahat$end_time = JASCO_malahat$UTC+ seconds(JASCO_malahat$dur)




JASCO_malahat$Provider = 'JASCO_Malahat'
levels(JASCO_malahat$Dep)<-c('STN3', 'STN4', 'STN5', 'STN6')

# Filepaths
dayFolderPath = 'E:/Malahat'
JASCO_malahat$FilePath =
  file.path('E:/Malahat', JASCO_malahat$Dep,JASCO_malahat$Soundfile)

JASCO_malahat$FileOk  = file.exists(JASCO_malahat$FilePath)


# Make sure all audio files are present for all annotations
if (all(JASCO_malahat$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}

# Nix annotations without audio files
""
JASCO_malahat= JASCO_malahat[JASCO_malahat$FileOk==TRUE,]

# runTests(JASCO_malahat, EcotypeList, ClassSpeciesList)

JASCO_malahat$CallType =JASCO_malahat$signal_type

JASCO_malahat = JASCO_malahat[,c(colOut)]

} else {
  JASCO_malahat <- data.frame(matrix(ncol = length(colOut), nrow = 0))
  colnames(JASCO_malahat) <- colOut
}





# # List which files are not in annotations list
#  audio.files = data.frame(
#    filename = list.files('E:/DCLDE2026/JASCO/Audio/',
#             pattern ='.wav', recursive = TRUE, include.dirs = TRUE))
#  audio.files$Soundfile =basename(audio.files$filename)
# # 
#  JASCO = merge(audio.files, JASCO_malahat, by ='Soundfile', all.x = TRUE)
# # 
#  JASCO$keep = JASCO$Soundfile %in% JASCO_malahat$Soundfile
# 
# # keep the file after all the start files, just incase
# idxExtraKeep = which(JASCO$keep ==TRUE)
# 
# JASCO$keep[idxExtraKeep+1] = TRUE
# 
# # Remove files without annotations
# filesRm = DFO_CRP[DFO_CRP$keep == FALSE,]
# file.remove(filesRm$filename)


#############################################################################
#University of Alaska data not organized by location. UAFdata.r used to
#organize at least into individual hydrophons or field locations

# New Audio Location (where to move the organized files too)
new_root =paste0(DATA_ROOT, '/dclde_2027_killer_whales/uaf/audio')

# Annotations location (already moved all the annotations here)
annot_root =  paste0(DATA_ROOT, '/dclde_2027_killer_whales/uaf/annotations/')

# Load the deployment info
UAF_depInfo = read.csv(paste0(DATA_ROOT, '/dclde_2027_killer_whales/uaf/metadata/Myers_DCLDE_2026_files.csv'))


# Get a list of annotation files
file_list <- list.files(path = annot_root,
                        pattern = '.txt', full.names = TRUE,
                        recursive = TRUE)

# Get a list of annotation files
file_list <- list.files(path = annot_root,
                        pattern = '.txt', full.names = TRUE,
                        recursive = TRUE)


# Read and concatenate the selection tables  with filename as a separate column (if non-empty)
UAF <- do.call(rbind, lapply(file_list, function(file) {
  
  # read the selection table
  data <- read.table(file, header = TRUE, sep = '\t')
  
  # get the audio file name
  audioFile<- basename(file)
  
  # Hydrophone id
  Hyd <- strsplit(audioFile, "\\.")[[1]][[1]]
  
  if (nrow(data) > 0) {
    
    # Add filename as a new column and get deployment from the filename
    data$Dep <- as.factor(strsplit(dirname(file),'/')[[1]][4])  
    data$Hyd <- as.factor(Hyd)
    parts <- strsplit(audioFile, "\\.")[[1]][1:2]
    filename <- paste(parts[1], parts[2], sep = ".")
    filename <- paste0(filename, ".wav")
    
    #AudioFile is one of the required column headings
    data$AudioFile =filename
    
    # For ecotype later on
    data$FolderName = dirname(file)
    return(data)
  } else {
    return(NULL)  # Return NULL for empty data frames
  }
}))

# remove 'table' from the field recordings 
UAF$AudioFile <- gsub("\\.Table1", "", UAF$AudioFile)
# and "table1"... shoot me in the face please
UAF$AudioFile <- gsub("\\.Table", "", UAF$AudioFile)


# Merge the files
UAF = merge(UAF, UAF_depInfo, by.x = 'AudioFile',
            by.y='FileName', all.x=TRUE)

UAF$FilePath<- file.path(paste0(DATA_ROOT, '/dclde_2027_killer_whales/uaf/audio'), tolower(UAF$Location), UAF$AudioFile)


# UTC is the UTC plus the file offset
UAF$UTC<- as.POSIXct(UAF$UTC)+seconds(UAF$Begin.Time..s.)

UAF$Soundfile<- UAF$AudioFile
UAF$AnnotationLevel<-'Call'


ClassSpeciesList = c('KW', 'HW', 'AB', 'UndBio')
AnnotationLevelList = c('File', 'Detection', 'Call')
EcotypeList = c('SRKW', 'TKW', 'OKW', 'NRKW', 'SAR')

# Ok Field recordings don't have an ecotype or population with them...Based
# on the filename Resident and Southern Alaska resident
UAF$Ecotype[is.na(UAF$Ecotype)]<- 'Resident'
UAF$finalizedEcotype =as.factor(UAF$Population)

levels(UAF$finalizedEcotype)<-c('TKW', 'TKW', 'OKW', 'SAR')
UAF$Ecotype = UAF$finalizedEcotype


############################################################################
# Finish up required headings
###########################################################################



UAF$HydId<-UAF$Hyd
UAF$HydId<-as.factor(UAF$HydId)
levels(UAF$HydId)[9:59]<-'Field'


UAF$Dep<- UAF$Location
# Two different hydrophones used in field deployments not recognized in the
# file names
nonFiledIdx = which(UAF$Dep != 'Field')
idxHTI = which(UAF$UTC< as.POSIXct('2021-06-15') & UAF$Dep == 'Field')
idxSoundTrap = which(UAF$UTC> as.POSIXct('2021-06-15') & UAF$Dep == 'Field')

UAF$Dep[nonFiledIdx]<- paste0(UAF$Dep[nonFiledIdx], '_', UAF$HydId[nonFiledIdx])
UAF$Dep[idxHTI] = 'Field_HTI'
UAF$Dep[idxSoundTrap]= 'Field_SondTrap'


UAF$KW = 1 # only KW annotated
UAF$ClassSpecies= 'KW' 
UAF$KW_certain = 1 # Only annotated calls that were certainly KW

UAF$Soundfile<- UAF$AudioFile
UAF$FileEndSec = UAF$Begin.Time..s.
UAF$FileEndSec = UAF$End.Time..s.
UAF$Provider <- 'UAF'
UAF$AnnotationLevel<- 'Call'
UAF$FilePath = UAF$audio_path
UAF$LowFreqHz = UAF$Low.Freq..Hz.
UAF$HighFreqHz = UAF$High.Freq..Hz.
UAF$FileBeginSec = UAF$Begin.Time..s.

# Change deployment to be the combination of the location
# and they hydrophone
UAF$temp = as.factor(paste0(UAF$Location,"_", UAF$HydId))
UAF$temp <- factor(UAF$temp, levels(UAF$temp)[c(2,3,4,6,5,7,9,8,10,1)])

UAF$FilePath = file.path(paste0(DATA_ROOT, '/dclde_2027_killer_whales/uaf/audio'), tolower(UAF$Location), UAF$AudioFile)
# Check that all files are found
UAF$FileOk  = file.exists(UAF$FilePath)


# Make sure all audio files are present for all annotations
if (all(UAF$FileOk)){
  print('All data present for annotations')}else{print('Missing data')}

UAF$CallType = ''

# runTests(UAF, EcotypeList, ClassSpeciesList)


UAF= UAF[,colOut ]


###########################################################################
allAnno = rbind(DFO_CRP, DFO_WDLP, OrcaSound, ONC_anno, scripps, SIMRES,
                VPFA_BoundaryPass, VPFA_HaroNB, VPFA_HaroSB, VPFA_SoG, SMRU,
                UAF,JASCO_malahat)

# Change 'Deployment' to dataset since towed arrays added
colnames(allAnno)[colnames(allAnno)=='Dep']<- 'Dataset'
rm(list = setdiff(ls(), c("allAnno", "OUTPUT_DIR")))
############################################################################

# Build a training subset
allAnno$ID = 1:nrow(allAnno)
allAnno$CenterTime = (allAnno$FileBeginSec+allAnno$FileEndSec)/2
allAnno$Duration = (allAnno$FileEndSec-allAnno$FileBeginSec)
allAnno$CalltypeCategory = 'None'
allAnno$HasQ = FALSE

# Figure out which data have unsure ecotypes and calltypes
allAnno$CalltypeHasQ = grepl("\\?", allAnno$CallType)
allAnno$EcotypeCertain = allAnno$Ecotype %in% c('NRKW', 'OKW', 'SRKW', 'TKW', 'SAR')

# Only go with certain species annotations in the PNW
KW_BaseData = allAnno[allAnno$Ecotype %in% c( '<NA>', 'TKW', 'SRKW' ),]
KW_BaseData = KW_BaseData[!KW_BaseData$CallType %in% c( "BP",
                                                        "buzz" ,"BZ",
                                                        "EL", "rasp", "W", "whistle",
                                                        "whistle/tone"),]
KW_BaseData = KW_BaseData[KW_BaseData$Provider != 'UAF',]
KW_BaseData = KW_BaseData[KW_BaseData$KW_certain==1, ]

# Add back in the not-killer whales
outAnnotations = rbind(KW_BaseData, allAnno[allAnno$KW ==0,])

outAnnotations = droplevels(outAnnotations)


# Create label's for the class/species datasets
outAnnotations$Labels = 'Background'
outAnnotations$Labels[outAnnotations$ClassSpecies == 'HW'] = 'HW'
outAnnotations$Labels[outAnnotations$Ecotype == 'SRKW'] = 'SRKW'
outAnnotations$Labels[outAnnotations$Ecotype == 'TKW'] = 'TKW'
outAnnotations$Labels[outAnnotations$Ecotype == 'SAR'] = 'SAR'

# Now we need to build a train/test split across the datasets paying particular
# attention to the abiotic and undtermined biologial sounds

#Fucking god dammed humpbacks
outAnnotations$CallType[outAnnotations$Labels == 'HW'] = ""




kw_cats <- list(
  TKW  = c("T01","T02","T03","T04","T07","T08","T11","T12","T13"),
  SRKW = c("S01","S02","S03","S04","S05","S06","S07","S08","S10","S12","S13",
           "S14","S16","S17","S18","S19","S22","S31","S32","S33","S36",
           "S37","S40","S41","S42","S44")
)

## Make sure CalltypeCategory exists and is initialized
if (!"CalltypeCategory" %in% names(outAnnotations)) {
  outAnnotations$CalltypeCategory <- "None"
} else {
  outAnnotations$CalltypeCategory <- "None"
}

## Loop over ecotypes in kw_cats
for (eco in names(kw_cats)) {
  toks <- kw_cats[[eco]]
  
  # indices for this ecotype (killer whales only)
  idx <- which(outAnnotations$Ecotype == eco & outAnnotations$KW == 1)
  
  if (length(idx) == 0) next
  
  for (i in idx) {
    ct_raw <- outAnnotations$CallType[i]
    
    # skip missing / empty
    if (is.na(ct_raw) || !nzchar(ct_raw)) next
    
    # normalise a bit (trim + upper) in case of inconsistent casing/spacing
    ct_norm <- toupper(trimws(ct_raw))
    
    assigned <- "None"
    
    for (tok in toks) {
      # match token in the normalised calltype string
      # here we do exact match OR pattern match depending on your naming style
      if (ct_norm == tok || grepl(tok, ct_norm, fixed = TRUE)) {
        assigned <- tok
        break
      }
    }
    
    outAnnotations$CalltypeCategory[i] <- assigned
  }
}


# Diagnostic-only block, removed: mapping_df was a CallType/CalltypeCategory
# duplicate-count summary that's never used by anything downstream, and it
# depended on dup_ids, which is never defined anywhere in this script.
#
# kwData = outAnnotations[outAnnotations$KW_certain ==1,]
#
# library(dplyr)
#
# mapping_df <- kwData %>%
#   mutate(
#     CallType = as.character(CallType),
#     CalltypeCategory = as.character(CalltypeCategory)
#   ) %>%
#   left_join(
#     dup_ids %>% mutate(is_duplicated = TRUE),
#     by = "ID"
#   ) %>%
#   mutate(is_duplicated = ifelse(is.na(is_duplicated), FALSE, TRUE)) %>%
#   group_by(CallType, CalltypeCategory, is_duplicated) %>%
#   summarise(n = n(), .groups = "drop") %>%
#   arrange(CalltypeCategory, desc(n))




# 1. Initialize HoldOut column
outAnnotations$HoldOut <- 'DCLDE_Train'
outAnnotations$HoldOut[outAnnotations$Provider == 'JASCO_Malahat'] <- 'JASCO_Malahat'

# 2. Filter for non-Malahat data
non_malahat <- outAnnotations %>%
  filter(Provider != 'JASCO_Malahat')

# 3. Select one row per file (assuming a column named 'File' identifies acoustic encounters)
file_info <- non_malahat %>%
  distinct(Soundfile, Dataset)

# 4. Stratified split (20% of files per Dataset)
set.seed(42)
test_files <- file_info %>%
  group_by(Dataset) %>%
  slice_sample(prop = 0.2) %>%
  pull(Soundfile)

# 5. Update HoldOut column for test files
outAnnotations$HoldOut[outAnnotations$Soundfile %in% test_files & outAnnotations$Provider != 'JASCO_Malahat'] <- 'DCLDE_HoldOut'


table(outAnnotations$ClassSpecies[outAnnotations$HoldOut == 'DCLDE_Train'])
table(outAnnotations$ClassSpecies[outAnnotations$HoldOut == 'DCLDE_HoldOut'])

table(outAnnotations$Dataset[outAnnotations$HoldOut == 'DCLDE_Train'])
table(outAnnotations$Dataset[outAnnotations$HoldOut == 'DCLDE_HoldOut'])

# -------------------------------------------------------------------
# Ensure each TKW calltype present in the data has >=1 example in TRAIN
# -------------------------------------------------------------------

# 1. What TKW calltypes actually exist in the annotations?
tkw_calltypes_present <- sort(unique(outAnnotations$CalltypeCategory[
  outAnnotations$Labels == "TKW" &
    outAnnotations$CalltypeCategory != "None"
]))

tkw_calltypes_present
# sanity check: this will NOT include "T13" if it never occurs in data

# 2. For each TKW calltype, check if train has at least one example.
#    If not, move one entire Soundfile from DCLDE_HoldOut -> DCLDE_Train
#    (so files still aren't split between train and holdout)

for (ct in tkw_calltypes_present) {
  
  has_train <- any(
    outAnnotations$HoldOut == "DCLDE_Train" &
      outAnnotations$Labels  == "TKW" &
      outAnnotations$CalltypeCategory == ct
  )
  
  if (has_train) next  # we're fine for this call type
  
  # Find holdout files with this calltype (non-Malahat)
  ct_hold_files <- unique(outAnnotations$Soundfile[
    outAnnotations$HoldOut == "DCLDE_HoldOut" &
      outAnnotations$Labels  == "TKW" &
      outAnnotations$CalltypeCategory == ct
  ])
  
  if (length(ct_hold_files) == 0) {
    message("No non-Malahat holdout files available for TKW calltype ", ct,
            "; cannot move one into training.")
    next
  }
  
  # Pick one file to move (first; or sample() if you want randomness)
  chosen_file <- ct_hold_files[1]
  
  message("Reassigning file ", chosen_file,
          " from DCLDE_HoldOut to DCLDE_Train to ensure TKW calltype ",
          ct, " is represented in training.")
  
  outAnnotations$HoldOut[
    outAnnotations$Soundfile == chosen_file &
      outAnnotations$HoldOut == "DCLDE_HoldOut"
  ] <- "DCLDE_Train"
}


# Check files are not repeated in train and hold out sections
DCLDE_train = outAnnotations[outAnnotations$HoldOut == 'DCLDE_Train',]
DCLDE_HoldOut = outAnnotations[outAnnotations$HoldOut == 'DCLDE_HoldOut',]
Malahat = outAnnotations[outAnnotations$HoldOut =='JASCO_Malahat',]

# The malahat data has a bunch of KW annotations that are not labeled
# to ecotype. These should be given an UNDKW label
Malahat$Labels[is.na(Malahat$Ecotype) & Malahat$KW==1] = 'KW_und'
DCLDE_train$Labels[is.na(DCLDE_train$Ecotype) & DCLDE_train$KW==1] = 'KW_und'
DCLDE_HoldOut$Labels[is.na(DCLDE_HoldOut$Ecotype) & DCLDE_HoldOut$KW==1] = 'KW_und'


write.csv(DCLDE_train, file.path(OUTPUT_DIR, 'DCLDE_train_parent_birdnetGrid2.csv'))
write.csv(DCLDE_HoldOut, file.path(OUTPUT_DIR, 'DCLDE_Holdout_parent_birdnetGrid2.csv'))
write.csv(Malahat, file.path(OUTPUT_DIR, 'Malahat_Holdout_parent_redo_birdnetGrid2.csv'))

# Check that files aren't split
any(unique(DCLDE_train$Soundfile) %in% unique(DCLDE_HoldOut$Soundfile))

