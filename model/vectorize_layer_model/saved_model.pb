ſ
? ?
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
=
Greater
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
m
StaticRegexReplace	
input

output"
patternstring"
rewritestring"
replace_globalbool(
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint?????????"serve*2.7.02v2.7.0-0-gc256c071bb28??
l

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1447*
value_dtype0	
}
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_97*
value_dtype0	
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
ǣ
Const_4Const*
_output_shapes	
:?N*
dtype0*??
value??B???NBtheBandBaBofBtoBisBinBitBiBthisBthatBbrBwasBasBforBwithBmovieBbutBfilmBonBnotByouBareBhisBhaveBbeBheBoneBitsBatBallBbyBanBtheyBfromBwhoBsoBlikeBherBjustBorBaboutBhasBifBoutBsomeBthereBwhatBgoodBmoreBwhenBveryBevenBmyBsheBnoBupBwouldBwhichBonlyBtimeBreallyBstoryBtheirBwereBhadBcanBseeBmeBthanBweBmuchBwellBbeenBgetBwillBalsoBintoBotherBpeopleBdoBbadBbecauseBfirstBhowBgreatBmostBhimBdontBthenBmadeBmoviesBmakeBfilmsBanyBcouldBwayBthemBtooBafterB
charactersBthinkBtwoBwatchBmanyBseenB	characterBbeingBneverBlittleBactingBplotBwhereBbestBloveBdidBknowBlifeBdoesBshowBeverByourBstillBbetterBoverBoffBendBtheseBsayBwhileBmanBhereBwhyBsuchBsceneBgoBscenesBshouldB	somethingBthroughBimBbackBthoseBdoesntBrealBwatchingBthoughBnowBactorsByearsBthingBdidntBbeforeBanotherBnewBnothingBactuallyBmakesBoldBlookBfindBeveryBsameBworkBfewBfunnyBgoingBusBlotBpartBdirectorBagainBcantBcastBquiteBthingsBthatsBwantBprettyBseemsByoungBgotBaroundBworldBfactBbetweenBdownBenoughBhoweverBtakeBgiveBbothBhorrorBmayBiveBthoughtBbigBownBoriginalBgetsBwithoutBalwaysBseriesBcomeBisntBsawBrightBlongBtheresBwholeBleastBtimesBmustBalmostBactionBpointBroleBfamilyBinterestingBbitBcomedyBmusicBdoneBmightBguyBscriptBlastBanythingBsinceBhesBfeelBfarBminutesBprobablyBperformanceBkindBamBratherBawayBworstByetBsureBtvBeachBwomanBplayedBgirlBmakingBfoundBourBfunBanyoneBhavingBalthoughBcomesBbelieveBtryingB
especiallyBcourseBlooksBgoesBhardBdayB	differentBshowsBputBplaceBwasntBbookBmainBonceBmaybeBendingBsenseBreasonBtrueB
everythingBworthBsomeoneBsetBmoneyBlookingB2BwatchedBactorBplaysBjobBscreenBdvdBtogetherBtakesBsaidBinsteadBseemBthreeBplayB	beautifulBeffectsB10BlaterBeveryoneBduringBhimselfBjohnBleftB	excellentBspecialBamericanBseeingBhouseBversionBnightBaudienceBshotBideaBsimplyBniceBwifeBblackBreadBhelpByoureBfanBstarBlessBusedBhighBkidsBdeathBsecondBelseB
completelyBwarBfatherBgivenBfriendsBuseBtryByearBpoorBperformancesBenjoyBneedBmindBwrongBhomeBmenBtrulyBeitherBboringBshortBrestBuntilB	hollywoodBclassicBnextBalongBlineBcoupleBtellB
productionBhalfBdeadB	recommendBrememberBletBstartBcameBothersBkeepBperhapsBfullBawfulB
understandBmomentsBgettingBcameraBepisodeBmeanB
definitelyBwomenBplayingBstupidBterribleBdoingB	wonderfulBstarsBsmallBsexBhumanBgivesBearlyBbecomeBoftenBperfectBnameBcouldntBvideoBfinallyBdialogueB
absolutelyBfeltBwasteBpersonBsupposedBcaseBlinesBfaceBpieceBlikedBtitleBlostBentireBitselfBschoolBtopBliveBbudgetByesBwrittenBagainstBwentBhopeBsortBshesB	certainlyBoverallBproblemBheadBworseBentertainingBstyleBseveralBpictureBbasedBwhiteBevilBboyB	beginningBcareBdarkBlovedBcinemaBfansBohBidBseemedBmrBexampleBlivesBmotherB3BalreadyBwantedBbecomesB	directionBdespiteBunfortunatelyBturnBchildrenBfinalBwontBguysBamazingB
throughoutBtotallyBkillerBfineBfriendB1BwantsBguessBlaughBsoundBhumorByoullBgirlsBlowBbehindBBmichaelBhistoryBcalledBturnsBdramaBleadBableBworksBtriesBsonBgaveBpastBqualityBdaysBunderBtheyreBwritingBfavoriteBgameBstartsBactBenjoyedBhorribleBkillBsideBtownBexpectB	sometimesBviewerB	obviouslyBdirectedB	brilliantBactressBstoriesBpartsBthinkingBgenreBcarB
themselvesBsoonBonesBeyesBfeelingBflickBdecentBartBbloodBheartBrunBsaysBfightBleaveBstuffBhighlyBcannotBillBmyselfBmatterBchildBexceptBtookBheardBhandBlateBkilledBwouldntBcloseBlackBhellBcompleteBparticularlyBmomentBcityBstrongBkidBpoliceBrolesBtoldBhappensBhappenedBhourBwonderB	extremelyBattemptBjamesBinvolvedB	includingBdaughterBetcBcomingBlivingBvoiceBshownBobviousBnoneBitbrBviolenceBsaveBhappenBlookedBscoreBpleaseBchanceBsimpleBmurderBanywayBgroupBseriousBtypeBtakenBageBslowBexactlyBcinematographyBaloneBletsBannoyingBsongBusuallyBgoreBbrotherBgodBstartedBhoursB
experienceBdavidBsadBusualBagoBokBinterestBwhoseBstopBhugeBhitBreleasedBendsBacrossBnumberBjokesBfindsBmusicalBknownBopeningBcareerBrunningByourselfBmostlyB	hilariousBrobertBcrapBchangeBrelationshipBfemaleBcoolBpossibleBcutBscaryBsomewhatBnovelBbodyBsayingBopinionBepisodesB	seriouslyBenglishBmajorBeventsB	basicallyBshotsBtalentBstrangeB
ridiculousBtalkingB	importantBorderBknewB5BwishBtodayBtakingBeasilyBpowerBheroBhappyB4BrealityBtellsB	directorsBcallBsingleB
apparentlyB
supportingBhusbandBroomBlevelBviewBdueBbritishBwordsBknowsBkingB	attentionBclearlyBlocalBarentBwhatsBdocumentaryBturnedBsongsBsimilarBproblemsBsillyBlightBearthBratingBpaulBfallsBjackBmoviebrBfourBwhetherBsetsBcomicBcheapBbeyondBwordBmodernBdisappointedBfiveBbringBmissBviewersBuponBrichardB	animationB
televisionBsequenceBgivingBgeorgeBfutureBrockBappearsBwithinBcountryBtalkBladyBactualBaddBreviewBpredictableBromanticBthemeBbunchBfeelsBlotsBredBfilmbrBpointsBamongBnearlyBneedsBhaventB	enjoyableBmentionBmessageBherselfBsequelBtheaterB	surprisedBtomB	storylineBparentsBmovingBaboveBmysteryBtenBteamBnamedBfallBeasyBtypicalBentertainmentBreleaseBdullBpeterBcommentsBworkingBtriedB	fantasticBelementsBsomehowBseasonBnearBcertainBbeginsByorkBwaysBavoidBmiddleBusingBdialogBstraightBeffortBfeatureBleadsBkeptBgeneralBfamousBwriterBthrillerBgreatestBbuyBweakBformBclassBhateBfigureBtaleBshowingBdoubtBfrenchB
particularBeyeBsorryBcheckBstayB	sequencesBspaceBmeansBlearnBclearBeditingBdecidedBsisterBgoneBfilmedBleeBwaitB	realisticBpoorlyBmaterialBlameB
soundtrackBhearB
eventuallyByouveBreviewsBoscarBdealBthirdBimagineBfastBwhosBmoveBdanceBbroughtBviewingB
atmosphereBsexualBkillingBtruthBdieBwritersBfollowBsitBdeBsurpriseBindeedBzombieBcrimeBwhateverBpossiblyBnorBsuspenseBsubjectBforgetBrentBexpectedB	emotionalBpremiseBnatureBjapaneseB80sBperiodB
filmmakersBstandB
screenplayBnoteBbabyBleavesB	difficultBbeginBneededBmeetsBaverageBbecameBokayBdramaticBromanceBdrBboysB	otherwiseBdisneyBquestionB	memorableBdogBrealizeB
believableBwriteBolderBreadingBfootageBstageBmeetBforcedB	situationBtotalBsuperbBcreditsBearlierBbringsBshameBkeepsBstreetBaskBwhomBunlessBsoundsBbadlyBweirdBplentyBfeaturesB
interestedBcreateBminuteBcommentBquicklyBdeepBfreeB
incrediblyBamericaBworkedBcastingBpersonalBcrazyBbeautyBtowardsBhotBplusBmaleBimdbBdevelopmentBvariousBsocietyBsettingBresultB	potentialBmeantB	directingBcreepyBbB	perfectlyBreturnBeffectBadmitB20BbattleBpreviousBhardlyBmessBmarkBdreamBcheesyBapartBleadingBmanagesBuniqueBremakeBopenBjoeBhandsBlaughsBbusinessBmissingBairBfailsBscifiBsecretBinsideBforwardBdumbBappearBunlikeBbrothersB	portrayedBideasBbillB70sBmonsterBlaBfantasyBattemptsBrecentlyBfireBfightingBfairlyBpowerfulB	christmasBoutsideBdeservesBtwistB
backgroundB	politicalBreasonsBwilliamBmasterpieceBfrontBjokeBsuccessBmatchBtalentedBpureBpayBpresentBcopyBcaughtBbreakBagreeBtellingBcuteB	followingBwesternBspentBmarriedBbenBrichBplainBmissedBjaneBcopBvillainB	expectingBsadlyBnudityB
incredibleBholdBflatBactedBgayBneitherBcomparedBdoctorBcrewBcreatedBfurtherBwastedBboxBdecidesBanimatedBslightlyBmembersBseesB
girlfriendBendedBcauseBsuddenlyBlargeBescapeBusesBpublicBfearBbooksBsweetBrateBpaceB	mentionedBzombiesBwaitingBislandBconsideringBintelligentBeraBpartyBoddBfamiliarBentirelyBbasicBsocialBtensionBdiedBvisualBlaughingBcoverB	audiencesBcreditBboredBconceptBcleverBwroteBscottBmovesBrecentBpopularBlistBviolentB	portrayalBmadBfilledB	effectiveBcartoonBvanBtroubleBpositiveB
convincingB
ultimatelyBcommonBexcitingBconsiderBlanguageBrevengeBwaterBspendBdancingBbiggestBbandB
appreciateB
successfulBdepthBchoiceBadultBspiritBkillsB12BscienceB	producersBmaryBformerBbizarreBofficeBcoldB	chemistryByoungerBspeakBgunBcatB	somewhereBstateBproducedB8BsolidBamountBcompanyB7BwonBfocusB	pointlessBgermanBwalkBsickBwerentBslasherBshowedBleavingBfitBvalueBrunsBfollowsBcontrolBwinBtonyBsingingB
situationsB
impressiveBamusingBrespectBfailedB	questionsBprojectBbarelyBtripBdecideBtoneBplanetBitalianBhairBawesomeBstoreB	involvingBimmediatelyBcenturyB
impossibleBtouchBsouthBrecommendedBstudioBasideBspoilersB
adaptationBthanksBlikesBhonestlyBchangedB15B
disturbingBsurprisinglyBimagesBgladBcharmingB	literallyBcampBprisonBjimBcollegeB	generallyBfrankBcultBvaluesBstarringBstandardBmagicBghostB
conclusionBlongerBfakeBknowingBaspectBforceBaccentBpatheticBnormalBshootingBsittingB30BstickBsteveBpicturesBcatchBboughtBnaturalBtrashBtoughBhonestBgeniusB	detectiveBcomediesBsexyB
personallyBabilityBmeaningBbeautifullyBwestBsmithBcomputerB
appearanceBwoodsBthinksBfairB
consideredBsoldiersBmanagedBsamBremainsBvampireBaliveBfullyBtwistsBsubtleBlondonBattackButterlyBtasteBexplainBchrisBarmyBappealByeahBplanBhumourBwalkingBpickBgarbageB	adventureBtermsBnowhereBmasterBexcuseBchannelBdadBthankBpassBmilitaryBlovesBpurposeBnobodyBterrificBoutstandingBchaseBsilentBrareBlovelyBequallyBcultureB100BtouchingBnakedBmoodBlikelyBselfBdreamsBdisappointingBcontainsByoudBbatmanBthusBthemesBslowlyBinnocentBcostumesBcomplexBthrownBpresenceBstunningBjourneyBdateBcryBcharmBanimalsBaddedBsurelyB
impressionBhopingBbottomBmannerBloudB
constantlyBchristopherBsentBmakeupB	laughableBwildBunbelievableBmistakeBtrainBmainlyBfictionBcharlesBroadBminorBharryB
governmentBbossBissuesBheyBedgeBdrawnBplacesBpainfulBbesidesBweekB	presentedBdetailsBstandsBnamesBmarriageB	cinematicBbruceBbrainBmakersBintendedBclimaxBvictimsBputsBphotographyB	exceptionBdoorB	narrativeBindianBheavyBthrowBpiecesB
mysteriousBemotionBawardBaspectsBstewartBfinishBfallingBcentralBbriefBsceneryBrideBratedBlawBjusticeBdisappointmentBclubBchurchB9BtrackBparisBmansB
historicalBexpectationsBtiredBsoulBshootBgangBcriticsBchangesBvictimBsuggestBcolorBsmartBfeelingsB	developedB
differenceBwowBsupposeBserialBlikableBhasntB
supposedlyBofferBfascinatingBactsBlaughedBfestivalBfilmingBtwiceBgiantBfollowedBelementBcreativeBbuildingBblueBtrailerBjerryBincludeBbedBemotionsBbotherB	availableBapproachBspoilerBfreshBopportunityBmoralBkeyB	confusingBmotionBlivedBimpactBsummerBspeakingBflicksB6BlacksBhotelBgorgeousB	everybodyBdiesBappearedB
thoroughlyBimaginationBhiddenBgradeBfellBsystemBprovidesBmillionBmediocreBflyingBconfusedBcharlieB	wonderingBsupportBnoirBmsBcontentBfellowBdriveBagentBadultsBnumbersBimageB	boyfriendBbornBrandomBhurtBdamnB
compellingBaheadBtimebrBshouldntBnoticeBlightingBzeroBstudentsBjonesBhenryBeventBdescribeB	seeminglyBputtingBproducerBnegativeB	happeningBbarBrentedBpainBpageB	impressedB	christianB	childhoodBshockBmurdersBmixBholesBgreenBanswerBalBshareBrelationshipsBoffersBmerelyBiiBprovesBdrugBdeliversBbillyBstuckBstepBlandBgemBforeverBfunniestBuglyB	standardsBlatterBkellyBgamesBdeliverBabsoluteBstudentBpaidBhelpsBextremeBraceBflawsBartisticBaddsBsixBloverBhospitalBheldBdeeplyB	americansBafraidBadditionBturningBrayBoperaBintenseBfolksBbondB	redeemingB	filmmakerBcompareBarthurBtragicBteenageBsoldierBremindedBpullBparkBinspiredBcarryBstruggleBmartinBsevenBsecondsBpickedBindustryBfindingBdetailBdesignBcountBtragedyBquickBdirectBbecomingBasksBstatesBjasonBfacesBaffairB90BskyBqueenBcriminalBalienB	actressesBthinBcaptainBwoodenBinformationBhumansB	favouriteBdyingBbrianBpersonalityBallowedB	technicalBstoneBprovideBmovedBmomBlordBgraceBunusualB	thereforeBsuperBloseBincludesB	forgottenB
collectionBcgiBwonderfullyBuncleBrapeBpornBcontinueBstephenBnastyBfoodBdouglasBcreatureB
attractiveBwarsB	scientistBdoubleB	desperateBangryBaccidentBwearingBremindsBrealizedBpassionBnewsBintelligenceBfashionBbeatBwilliamsBsuperiorBontoBjoanBholdsBgroundBclichéB
whatsoeverBtrustBrussianBreadyBepicBworthyBweddingBallowBspotBmentalBareaB
remarkableBnormallyB	nightmareBledBdrugsB	accordingBtimBshockingBplaneBlocationBlistenBhelpedBbeganBringBprofessionalBfoxBdannyBtreatB
introducedBextraBwillingBmartialBindependentBgrandBdirtyBcopsBcomedicBscaredBdavisBchineseBanimalBallenBteacherBpowersBpleasureBdesireBbuildB	necessaryBmemberBgrowingBapparentBtodaysBtearsBliesBenergyBacademyBwallBunnecessaryBtheatreBsatBcastleBanymoreBwarningBtowardBshipBsearchBmonstersBmachineBhatedBclichésB	apartmentBacceptB60sB50BwantingBtortureBskipBloversBhigherBcaptureBanywhereBsuspectBrarelyBphysicalBexplanationBdisasterBdickBclothesBalanBsmileBmouthBedBconstantBartsBsleepBcreatingBactionsBlegendBladiesBjohnnyB40BreturnsBnicelyB
intriguingBheroesBdonBbitsBartistBvsBlimitedBkillersBgunsBengagingB
commentaryBbloodyB	religiousBjoyBandyB	watchableBvillainsBstationBsightBmemoryBmediaBjumpBfinishedBsomebodyBphoneB
originallyBmoonBieBfinestB
filmmakingBfailBaskedBabsurdBwinningB
surprisingBpacingBjrBinstanceBhopesBheresBbrutalBblameB
delightfulB	dangerousBcarsBanybodyBwitchBvisionBrollBplayersBmetBlackingBkevinBjeanBaccurateBsuitBlovingBkeepingBenglandBunknownBprocessBjapanBhitsBwheneverBstartingBproveBmorganBmonthsBkindaBheavenBgeneBeddieBpeoplesBfaithBwerewolfBteenBsavingBregularB
friendshipBdeserveBprinceBportrayBordinaryBmixedBissueBworldsBsavedBpilotBgottenBeatBconflictB	communityBheadsBfightsBsupermanBquietBprivateBnickBlosesB	featuringBdiscoverBwhilstBvampiresBtaylorB	screamingBriverBmanageB	lowbudgetBfatBexistBcageBcableBweveBthembrBterriblyBsuicideB	reviewersBpulledBfredB
washingtonBspanishBpsychologicalBpriceBplotsBmemoriesBhadntBdatedBunderstandingBtreatedB	explainedBcutsBcBwittyBunfunnyBtraditionalBsucksBseanBresponsibleBrecordBrealismBjesusBheroineBessentiallyBdeservedBbiggerBadamBofficerBgaryBdrunkBdrivingBbrightBanthonyBlooseB	knowledgeBemptyB
connectionBawareB1010BpartnerBoppositeBnumerousBjacksonBiceBhumanityBfieldBeuropeanB
discoveredB	continuesBbrokenBblandBvisuallyBvisitBpopBopensBmissionBmagnificentBlengthBfranklyBfateBdealsBcuriousBcapturedBblondeBwindBvhsBplayerBlossBgenuineBforcesBallowsByouthBsuddenBsoapBradioBpretentiousB	naturallyBkateBstreetsBreceivedB	daughtersBcurrentBbobBvillageBspectacularBskillsBprovedBnonsenseBmurderedBmrsBmikeBjudgeBultimateBthisbrBsegmentBownerB	locationsBinternationalBeffortsBbreaksBstockBsingBrobBharrisBgoldBcornyBcallsBversionsBsignBrubbishBresultsBnationalBmorningBkongBjimmyBballBadviceBwiseB
unexpectedBtheyveBtalentsBstandingBlessonBblindBwindowBvisualsBnoticedBjeffBhimbrBfloorBdealingBbelowBwellesBsatireBsantaBpleasantBhumorousB	concernedBannaButterBstudyBstealBsiteBscreamBrevealedBprogramBoccasionallyBlogicBfordBcreatesBbrilliantlyBreactionB	presidentBperspectiveBpairBluckBfathersBcausedB90sBsurviveB	discoversBdevelopBawkwardBalbertBshopBlearnedBgrantB	genuinelyBfamiliesBdressedBdebutBakaB1950sBmineBmilesBincludedBgrewBfreedomBcameoBbehaviorBthomasBreachBoverlyBleaderBhorseBdanBcontextBcenterBboardBawardsBalexBunableBtypesBsuckedBsheerBshallowB
referencesBmajorityBgonnaBgagsBfavorBcrappyBcandyBbuddyB
technologyBstevenBspeedBseasonsBformulaB	deliveredBunrealisticB
theatricalBstereotypesB
rememberedBreligionB	meanwhileBkeatonBgraphicBgangsterB	existenceBaskingBwitB
underratedBseaBpatrickBparkerBfaultBbuiltBbetBassumeBagesBwoodBstronglyBspoilBsakeBreviewerBpracticallyBneverthelessBforeignBericBdestroyBdecadeBdanielBcrossBbelievesBtarzanBrobinBrelateBproduceBlargelyBhearingBgoldenBcombinationBchooseBbankBtravelBprotagonistB	painfullyBlosingBlaughterBjenniferB
gratuitousB
flashbacksBbrownBvictorBukBsheriffBriseBrangeBproperBfootBeveningBeditedB	contrivedBtapeBstrengthBsellBrulesB
portrayingBgrowBfinaleBdrewBdecisionB
comparisonBclueBbarbaraBauthorBwoodyBtwentyB	sufferingBsingerBreliefBnaiveBdesertBdBcostsBcapableBstoppedBsequelsBroseBpassedB	hopefullyBhitlerBfeetBembarrassingBdreadfulBcontrastBchosenBbearBtrekBtestBranB
generationBfootballBendlessBdevilBbroadwayBvoteBtrainingBthoughtsBsistersBsimonBmeetingBmattersBlouisBgermanyBfillBfailureBextrasBbombBattitudeBasianBancientBvehicleBruinedBrBpostBparodyBmarryBluckyBlowerBlakeBinsaneBexecutedB
depressingBannBwalksBtalksBstorybrBsportsBrescueBnativeBidentityBheckBgoryBentertainedBeatingBlearnsBjosephBframeBclassicsB
cinderellaBanneBwideBunlikelyB	teenagersBportraysB
individualBhorriblyBallbrBtendBscareBrainBproductBhillBfareB	creaturesBcostumeBbaseballBanimeB	treatmentBsoftBroundBresearchBlawyerBinsultB	halloweenBfitsBfactsBexcitedB	describedBdepictedBchiefBwinnerBtitanicBsympatheticBonebrBmebrB
irritatingBinvolvesBgrownBfreddyB
excitementBemotionallyBcloserBcleanBchickBcapturesBtinyBthatbrBservesBlukeBlevelsBhowardBdryBbodiesBboatB
amateurishB810B50sBtheatersBhandsomeBexploitationBdogsB
commercialBclaimBchoseBbringingBwalterBunitedBtillBsupernaturalBsonsB
satisfyingBkickBhunterBhauntedBgordonBevidenceBcrowdBcartoonsBwellbrBvoicesBsuffersB	substanceBsendB
relativelyBlewisBinitialBguiltyBfoolBfleshB	appealingBanglesBafricaBwarmBtediousBsaturdayBpromiseBpriestBobsessedBnorthBmaxBjackieBhalfwayBdangerB11BstudiosBreporterB	qualitiesBmodelBmattBmaskBhangingBgrantedBeuropeBasleepB1970sBtargetBrogerBnancyB
mainstreamBhideBflyB
disgustingBconvinceBcanadianB	virtuallyBvictoriaBtouchesBshockedBroutineBrepeatedB
previouslyBoliverBlatestBhauntingBfameBexperiencesBenemyBcostBcashBcaresBappropriateBaccidentallyBweeksBstealsBseatBsBryanBremainBrecallB	promisingBpityBoutbrB	offensiveBmindsBlynchBdeadlyBcategoryBamateurBwayneBstorytellingBpresentsBhandledBdrawBdragBdeathsBcoveredBwelcomeBweekendBtrilogyB
strugglingBspeechBsharpBrevealB	professorBholdingBfocusedBdubbedB
continuityBcircumstancesBbreakingB
adventuresBwalkedBsurrealBsourceBsafeBruinBroyB
overthetopBinnerBenterBdesignedB	convincedBcolumboBwouldveBspeaksBshakespeareBserviceBremotelyBpullsBprovidedBpileBherebrBharshBcontemporaryBcolorsBclaimsBbugsBveteranBunfortunateB	surprisesB	structureBrentalBpsychoBmovementBmassiveBhallBfranceBcorrectBcoreB710BsinatraBrobotB	recognizeBplansBmistakesB	melodramaBmarieB	invisibleBfBexplainsB	downrightBassBangelBaliensBaccentsBviewedBuniverseBtwistedBringsBprincessBpowellB
occasionalBmultipleBlesbianBkindsBfridayBdegreeBwhoeverBirishBfairyBexpressBedwardBdirectlyBcombinedBbotheredBblowBbirthB	atrociousBtreasureBteensBteenagerBspendsBprimeBpoliticsB	performedB	nominatedB	narrationBlousyBlonelyB	highlightBhatBfalseBfactorBdropBconversationB	childrensB
australianBartistsBvietnamBuninterestingBsuspensefulBsecurityBrussellB
propagandaBmexicanB	legendaryBidiotBhuntBfrighteningBeightBchangingBbraveBandersonB1980sB	statementB
refreshingBmarketB	influenceBhongBghostsBforestBemmaBdisplayBdarknessB	committedBwitnessBwilsonBsympathyBpriorBpathBmagicalBhedBfuBfiguresB	fictionalBangerBwaybrB	universalBspyBpaintBofferedBnetworkBmgmBfiguredB	executionBdollarsBdeliveryBdeeperBandorBaliceBunconvincingBtrappedBsurfaceBrequiredBpaperBmothersBmooreBjunkBjobsBjBisbrB	favoritesBeffectivelyBdesperatelyBvonBtheoryBsundayBsleepingBregretBprintBjuliaBjonBfeaturedB
everywhereBdonaldB
departmentBdemonsBclarkB13BtexasBsectionBpeaceB	listeningBkimBhiredBcowboyBamazedBafricanB25BwearBvarietyB
thankfullyBscaresBroughBnudeBlifetimeBlearningBgraveBexperiencedBendbrBcuttingBcrudeBweaponsBsummaryBreminiscentBrealizesBrachelBpittBpassingBmexicoBmatureBmatrixB	interviewBheavilyBfourthBexactBdozenBcriticalB	criminalsBcrashBburnsBbmovieBaccountB14BsoundedBnecessarilyBmountainBintellectualBinsightBimageryBexamplesBdeviceBabuseB	abandonedB1930sB
worthwhileBstayedBskinBsignificantBseekBquestBpayingBpacedBmariaBlightsBjulieBfreemanBforgotBfocusesBeroticBcodeBclosingBclichédBcaineBbelovedBbelievedBachieveBtedBtechnicallyBsuckB	subtitlesBsortsBsirBscaleBrevealsBpacinoBgrimBforthBforgettableBdragonB	depictionB
californiaBusaBteethBregardBrawBpreferBplacedBmusicalsBfriendlyBfacialBextentB
expressionBdriverBdeanBcarryingB110BspiteB	slapstickBsarahBripBrelatedBmildlyBlucyBhelpingBgasBfunnierBcourtBchinaBblahBaintB
understoodBtableB	strangelyBstolenBstaysBsettingsBscenarioBpregnantB	ludicrousBignoreBgrittyBgreaterBfabulousB	encounterB	elizabethBdrivenBcryingBconveyBbusBblockbusterBangleBagainbrBviaBurbanBtouchedBsusanBsucceedsBstarredB	sensitiveBrightsBproductionsBonbrBnovelsBinspirationB	flashbackBfaithfulBextraordinaryBexpertB	entertainBcarriesBcaringBbuyingBbeliefBangelsB	amazinglyBstopsBskillBserveBnavyBlazyBkungBkissBfallenBembarrassedBcomicalBboreBbaseBamongstBviewsBundergroundBsubplotBstereotypicalBsidneyBsanBrogersBpositionBlugosiBlieB	disbeliefBcontainBcampyBanswersBtrickBsitcomBraisedBmetalBlockedBlifebrBjoinB	happinessBfbiBdudeBcruelBcallingBattacksBsinisterBshadowBronBrollingBproudBprotectBpraiseB	notoriousBnonexistentBlauraBlarryBgraphicsBenvironmentBdecadesBcouldveBbreathtakingBblairBtalesBstanleyBpicksBnedBmurdererBironicBhenceBformatBeverydayBdescriptionBdailyBbridgeBbreathBbradBalrightB310BwinsBwarnerBupsetB	traditionBterrorBsufferBrentingBraiseBpurelyBproperlyBpreparedBoscarsBmurphyBmanagerBleslieBinternetBineptBindiaBgoofyBcomplicatedBcellBbusyBbasisBarmsB	afternoonBwesternsBwearsBwaveBwarnedB
revolutionB
reputationBquirkyBprotagonistsBmereB	marvelousB	inspectorBhanksBgoodbrBgodsBfosterBflightBenjoyingBdrinkingBdemonBculturalBchillingBblownBbeastBashamedBwriterdirectorBthrowingBstBruleBrivalBratingsBpunchB	obsessionBnotableBglimpseB	challengeBauntB410BxBvacationBtoyBthrowsBtaskBswordBsunBsoldBsocalledBservedBriskBretardedB	regardingBpetBopenedB
interviewsBhorrificBguardBgrossB	destroyedBdennisB	criticismBbeachB2000BtimingBscriptsBremindBoriginalityBjungleBjailBdislikeBcausesBcarriedBcabinBbourneBbbcBattackedBwastingBtruckBtitlesBshutBlaneB	initiallyBindieBhonorBgruesomeBdrivesBbalanceB	authenticBappreciatedBaddingBtenseBspoofB	referenceB	obnoxiousBnowadaysBlyingBleagueB
experimentBcurseBchoicesBchargeBcasesBarrivesBandrewsBturkeyBtracyBtightB	thrillingB
sutherlandBstylishB	strugglesBsouthernB
scientistsBpresentationBovercomeBmichelleBlionBlesserBjohnsonBjessicaB	innocenceBidioticBhusbandsB	hitchcockBhaBguessingB
frequentlyBflawedBdressBdelightBclipsB2006BwwiiBtreeBtopicBtermBsuccessfullyBrefusesB
meaningfulBintroductionBhoodBfortunatelyBcrisisBbitterBwarnB	stupidityBstrangerBsingsBrobertsBreplacedBremoteBjumpsBhundredsBhintB	essentialBcontroversialBbuckBbettieBalbeitBwishesBtornBsullivanBstomachBstatusB	searchingBracistBpoignantBneedlessBmillionsBmentallyBmassBmadnessBinterpretationB
intentionsBhardyB	franchiseBescapesBcynicalBclaireB	bollywoodBatmosphericBseekingBportraitBoddlyBmouseBmidnightBlegsBkurtBflowBdinnerBcontactBbrokeBairedB910BweaponBuncomfortableBtextBsuspectsBstepsBshowerBseparateB	screeningBripoffBridingBproofBphilipBmirrorBgloryBglassBcornerBcomedianB	thousandsBsuggestsBsidesBshedBrocksBrevolvesBreedBracismBperformBpatBnationBmiscastBlucasBjayBhelenBexpressionsB	enjoymentBegBeastBcookBbadbrB
attemptingB80BtroubledB	thrillersBtherebrBstormBsinBshortsBplasticB
physicallyBpatientBnineBmonthBlovableBlessonsBjewishBhandleBexistsBdollarB
determinedBbatBwomansBwhereasBwealthyBuB	techniqueBsufferedBspotsBspokenBshinesBsavesBnonethelessB
incoherentBhorsesBhidingBhappilyB	connectedBbrooksBzoneBwidmarkBtriteBprideBpersonalitiesBoBneighborhoodBmansionBhillsBhamletBdevelopsBcredibleB	competentBchristBbagBachievedBthiefBsumBslightB	rochesterBreunionBrepeatBralphBnotedBmillerBhundredBfactoryBdubbingBdareBcourageByoubrBtuneBtrialBstoodBstealingB	sexualityBpersonsBparBpackBobjectBnaziBhostBholidayBhoffmanBgutsBgreatlyBdigBcrimesBconsistsBcloselyBcheckingBchasesB	cardboardBcameosBbetteBbelongsBattachedBappearancesBandrewB	advantageBacceptedB30sBworryBtributeBteachBspikeBsnowBsleazyBnoseBnobleBnelsonBmindlessBlisaBlibraryB	intensityBhomelessBgrippingBfortuneBensembleB	endearingBdorothyBcolorfulBchasingBcatsBbetterbrB	believingB
attractionBadaptedBwallsBunintentionallyBstretchBscreenwriterB	ourselvesB	onelinersBnBmoviesbrBmobBlloydBinfamousBhuntingBhB	expensiveBdraggedBdialogsBdenzelBdearBcubeBcomicsBannoyedBstrikingBshootsBoilBluckilyB	godfatherBgainBfishBestablishedBentryBeasierB	complaintBvideosBtiedBsucceedBshapeBsentimentalBsegmentsBrippedB	reactionsBpoolBmedicalBlaidBkingsBimaginativeBianBgagBflawBexceptionalBdealtB	countriesBcivilBcharismaticBburtBbearsB	attractedB18BwakeB
terrifyingBtBsubplotsBsandlerB
performersBmst3kBmassacreBjeremyB	intentionBholeBherbrBcoversBcooperBconcertBconcernsBcheeseBchairBbranaghBboundBarmB	appearingB1960sBunforgettableBtourBtoobrB
techniquesBsurroundingBstanwyckBsetupBsendsBprofoundBpieBironyBhorrorsBhangBgreyBfxBeerieBcatholicBcardBboBwannaBuselessBstuntsBstrongerBshallB
revelationB
redemptionBpitchB
perfectionBpaintingBnuclearBneatBmatthauBkarloffBjakeBheartsB
frustratedBessenceB
encountersB	confusionBcharismaBchanBbirthdayBalasBagedBwonderedBstrictlyBreynoldsBrelevantBquoteBnotchBnightsBlosBknifeBhandfulBguestBgloverBequalBdroppedBdramasBdawnBcredibilityBcorruptBcontractB	continuedBcolourBchuckB	carefullyBburiedBbeatingBbattlesB	assistantB1stB17BwarriorBvirginBvincentBtricksBstrikesBspendingBseagalBreallifeBpleasedBplagueBneckB
miniseriesBmenacingBmagazineBlettingBletterBitalyBhudsonBgothicB	countlessBbrandB	attemptedB
acceptableB24BwinterB
universityBunionB
surroundedBspecificBsophisticatedBsomeonesBsilverBsilenceBrapedBneighborBloserBlincolnBkoreanB	intriguedBiiiBhopedBgoalBgiftBfittingBfiredBfilmsbrBfifteenBevaBdoorsBdisagreeBdentistBdancerB	conditionBchancesB	catherineBboredomBborderB
associatedBaccusedB40sBtradeBthousandBshiningBpsBpointedBlistedBlackedBkenBindividualsBimportantlyBgrowsBgottaBflawlessBcurtisBcanadaBbobbyB	ambitiousBworkersBweightBtheydBrelativeBpushedBnazisBkicksB
importanceBhookedBhomageBflashBfayBfacedBdocumentariesBcraftedB
conspiracyBcharacterizationBbucksBboxingBbollBallowingBaforementionedBadamsB	worthlessBvirusBtonsBtheyllB	territoryBtalkedBsuperblyBsticksBsolveBrequiresB
outrageousBnotablyBmustseeBmileBmediumBmarksBkillingsBhipBelvisBdrinkBcompetitionBcommitBcloseupsBcarolBcamerasBbulletsBbeingsBbasementB	appallingB
afterwardsBwickedBvagueB
uninspiredB	typicallyBstatedBsplitBspellBshortlyBshadowsBsexuallyBrushedBrowBprojectsB
presumablyBpressBperB
overlookedBoccursBleB	kidnappedB
horrendousBhireBhealthB	graduallyBevidentBdukeBdrawsBdestructionBcreatorsBcheBcaredBblowsBbarryBachievementB3dBwondersBwalkenBtrapBthumbsBspoiledBsoulsBrobotsB	revealingBrealiseBpreciousBpartlyBmessagesBlargerB	insultingBignoredBidentifyBhestonBgrayB	elsewhereBdragsB
discussionBderekBcousinBcoleBbrieflyBbrandoB2001BwatersBwardBunbelievablyBtoiletBthrillsBsunshineBstylesBstrikeBstanBsoleB
resolutionB
regardlessBprogressBpreventBplanningBpersonaBmelodramaticBjumpingB
highlightsBgoldbergBflynnBfearsBfatalBemBelderlyBdisplaysB	curiosityBcupB
convolutedBburningBbritainBwingBsuitsBstuntBshyBsallyB	returningB
reasonablyBpianoB
performingBopposedBmildBmeatBmafiaBkaneB	inspiringBimprovedBhuhBgenerousBdocB
cameraworkBburnBbrideBbeerB2005BwesBwatchesBwannabeB
subsequentBspringBspinBrushB	remainingBpushBpickingB	overratedBnurseB
motivationBmoralityBlabBknockBinvestigationB
inevitableBincreasinglyBguiltBgrandmotherBgenresBexerciseBestateBempireB
depressionBcruiseBcameronBbeatsBatlantisBargueB	alexanderBaimedB911BtwelveBthreatBsplendidB	spiritualBsmoothB	secretaryB
scientificBsadisticBresponseB
repetitiveBrapB	providingBphotographedBnotesBnoiseBminimalBmidBkapoorB	instantlyBhittingB	franciscoBexplicitBemilyBdrawingBdoctorsBdirectorialBdarkerBdanesBcitizenBbleakBargumentBaffectedBabsenceB45BtimelessBticketBsloppyBsavageB	sacrificeBrootB
representsB
repeatedlyBnonB	mountainsBmelBmatthewBlogicalBjerkBintentBidiotsBhatredBfancyBentersBdvdsBdiseaseB	dinosaursBcriedBconsequencesBcluesB	carpenterB
brillianceBbeatenBastaireBarrestedBadmireBadequateBwarrenBturnerB
tremendousBtorturedBtimothyBthugsBstiffB
spoilersbrBsmokeBscreamsB	resemblesBpushingBposterBkudosBjealousB
ironicallyBgreekBfunnybrB	forbiddenBfarmBfailingBdistanceBdiamondBcraftBcaptivatingBburnedBblankBbellBagingB210BthrewBtallBsubjectsBstruckB	slightestBshellBrichardsB
restaurantB	representB
reasonableBreachedBpurpleBpsychiatristB	producingBpackedBoughtBofficialB	miserablyB	manhattanBlegBkennethBjuvenileBjeffreyBhypeBhatesBguideBgentleBfeverBfarceBexploreB	executiveB	elaborateBdireBdevotedB	currentlyBcubaBconversationsBcharactersbrBburtonB	broadcastBbrazilBblendBbakerB	australiaBalltimeBalikeBadBupbrBtreesB	symbolismBsovietBreliesBreactBranksBpovertyBperryB
overactingBneilBmonkeysBmodestyBmastersBlikingBliberalB	landscapeBkitchenBindiansBhousesBgroupsBgadgetBforgiveBextendedBdraculaB	dialoguesBcoxB
concerningBcomplainBarnoldB
admittedlyBworldbrBwintersBswearBsuperficialBsignsBsellingBselfishBritterB	psychoticBproceedingsBobscureB	movementsBmitchellBmissesBlooselyBloadBinvolvementBinstallmentBhughBheroicBharderBhardcoreBglennBedieBdrunkenBdozensBdistantBdefiniteBcomfortableBclothingBciaBchildishBbrosnanBbrainsBblobBbibleBaudioBarriveBareasBwreckBwebsiteBupperBtommyBthroatBswedishBstoleBspiritsBslapBscriptedBsamuraiBsadnessBreturnedBreducedBpretendBoverwhelmingBofficersBoccurredBmeritBlilyBjamieB
innovativeBincidentBhurtsBholmesBhighestBheatBgandhiB
futuristicBformsBescapedB	eccentricBdetailedBdemandsB	decisionsBcatchesBcaryBbroadB20thB	yesterdayBvividBunintentionalBtwinBthirtyBsurfingBsecondlyBresemblanceBreceiveBpullingBphotographerBoverdoneBordersBoccurBmeltingBmaggieBmachinesBimpressBhorridBfocusingBexploredB
explainingBexistedBenormousBdynamicBduoBdrivelB	discoveryBdigitalBdancersBcreationBcraigBcarreyBbuddiesBbradyBaprilBworriedBworkbrBwBviciousBvastBupsBthrillBthreateningBtempleBstripBstayingBshipsBsecretsBramboBpopcornBpetersBofferingBnormanB
journalistBintrigueBimplausibleBimaginedB
hystericalBhollowBgardenBfighterBexBeveBetBenjoysB
developingB	describesBdeliberatelyBcouplesBcorpseBchoreographyBchaplinBcarlBcagneyBbuildsBbrooklynBbreastsBbeneathBbathroomBadvanceBaccomplishedB2003B16ByellowBwolfBwebBunevenBtunesBtriumphBtrioBtranslationBtitledBthickBtameB	survivorsB	superheroBstillerBspookyBsmokingBsizeBruralBremovedB	realizingBpulpBpromB
pleasantlyBpassesB
nightmaresBmeasureBmakerBlaurelBjetBirelandBinvolveBgenericBfalkB
enterpriseBeditorBdiscussBdignityBdancesBcontraryBcommercialsBchasedBboldBblewBbiteB2004BwidowBwealthB
unpleasantBtransformationBtoysBtoddBteaBstringBsnlBpossibilityBpossibilitiesB	possessedBpanicBlightheartedBkarenBinvestigateB
introducesBholyBhitmanBfurthermoreBfuneralBfolkBfloridaBexoticBexaggeratedBengagedBelviraBdoomedBdifferencesBconnectBcombatBchainBcgBblatantBbedroomBapesBanywaysB
altogetherBwatchbrB
unbearableBtiesBstressBsoloBruthBridiculouslyBprimaryBpaysBmodelsB	madefortvB	inventiveB	illogicalB
hollywoodsBfisherBetcbrBdianeBdevoidBcusackBconsistentlyBcarterBbenefitBbannedBbackgroundsBbackdropBarrogantBamyBahB2002BurgeBunwatchableBtravelsBtenderBstagedBstaffBshoesBshineB
recognizedBreachingBpreviewB	neighborsB	murderousBmethodBlatelyBkingdomBjonathanBhammerBguessedBgBeyreBexposedB	explosionBendingsBdistractingB
disjointedBdestinyB	depressedBconventionalBcomposedBchorusBcausingBbinBannieBangelesB60ByellingBwivesBtimesbrBtapB	spielbergBshowbrB	seventiesBsettleBschemeBscenebrBsagaBruthlessBpagesBnotionB	nostalgicB	newspaperBmickeyBlipsBjudyBjesseBinstantBhollyBhalBfliesBemphasisB	disturbedBdisappearedBdefeatBdaringBdamageBcrushBclintBcheatingB	charlotteBchaosBbirdBbandsB
artificialBanticsBabusiveB2ndB0BversusBtoplessBterryBsynopsisBsportBsoccerBsmallerBshelfBsharkBseenbrBromanB
remarkablyBquotesB	purchasedB	primarilyB
populationBphotosB	performerBoutcomeBoddsBnearbyBminiB	lifestyleBjudgingBillegalBhartBgrabBgloriousBflopBeaseBdropsBdollB	displayedB	dedicatedB	containedB	commentedBcoherentBcoffeeBcareersBbushBbirdsBadorableB1990sBwalkerBuserBunfoldsBtadBswitchB	streisandBstellarB	similarlyBsidekickBsatanBsafetyBroomsBridBrebelBpurchaseBpropsB
philosophyBouterBonedimensionalBmotivesBmoodyBmixtureB	miserableBmildredBmenaceBmeaninglessBlouBlikewiseBjustifyBignorantBidealBhboB	gangstersBfrustrationBfondBearnedBdiscBdesignsBclumsyBbluesBavoidedB1996BwedBweakestB	wanderingBuweB	travelingBtrailersBtonightBthruBsurvivedBsnakeBslaveB
simplisticB	senselessBsellersB	satisfiedBreportBrageBprequelBpoliticallyBpaintedBonscreenBnicholasB	mysteriesBmonkeyBleonardBlayBkirkBisolatedBhuntersBhonestyBhintsBgodzillaBgingerBengageBendureBegoBeBdubBdivorceBdatingBcountrysideB
corruptionBconsiderableB	comparingBclaimedBcentersBbuffsBbuffBakshayBaffordBabcBunhappyBunderstatedBunderstandableBtroopsBtracksBswimmingBsuitedBsubtletyBstreepBsimilaritiesBscopeBrepresentedBrecordedBreachesB	principalB
passionateBorsonBoffendedBmargaretBmarchBlyricsBkiddingBjumpedBinteractionB
influencedBinferiorBheartwarmingBgraspBfreakBerrorsBellenBdutchBdisneysBdirectsB
deliveringB	corporateB	companionBcircleBbottleBblockBblacksBaweBagentsBaffectB	abilitiesB3rdByaBwishedBwillisBwallaceBvirginiaBtrashyBtransferBsubBstinksBsplatterBspareBsitsBrobbinsBrivetingBremarksB	relationsBrankB	provokingBpromisedBpigBparentBnervousBnarratorBmontageBmarioBlinkBlawrenceBkhanBingredientsBhideousBheadedB	formulaicBfingerB
consistentBcliffBcleverlyBclassesBcinematographerB	celebrityBcaveBbonusBaidBadvancedB70B1972BwomensBwindsBwaitedB
vulnerableBvoightBunawareBtongueBtearBsurvivalBsuitableBstiltedBsteelBstaringBsolelyBscottishBscoresBscenesbrBrocketBrevolutionaryBplantBozBmotivationsBmistakenBlindaBleesBlastedBjoeyBitllBimproveBhungBhumbleBfeedBdollsB	disappearBdaviesBdaddyBcommandBcolonelBclosetBcardsBbebrBamandaBaccompaniedBwasbrB
unsettlingBtiresomeBquestionableBpornoBplightBphilosophicalBoutfitBoccasionBmyersBmorebrBmiikeB	macarthurBlustBlivelyBlatinBkickedBinbrB	immenselyB
horrifyingBhookBhomerBhamiltonBgrinchBgrandfatherBgialloBfloatingBfelixBexplorationB	excessiveB
destroyingBcriticBcrackBclosestBcitiesBchicagoBchestBcheatedBchallengingBcampbellBbuttB
basketballBarguablyB	alongsideBagreesBagreedB1999BwavesBtieB	terroristBstinkerBspreadBspecificallyBsimmonsBshirleyBshakeBsappyBrickBresortBnotbrBniroBnamelyBmuchbrBmontanaBmatchesBmaintainBloadsBleoBjazzBinaneBhandedBgoodnessBfrancisBfaultsB
explosionsB	everyonesBearlBdesperationBderBdefendBdaveBconvincinglyB	communistBcoachBcatchyBbulletBbrutallyBbakshiB	attitudesBamountsBadviseBworeBvBundoubtedlyB
simplicityBsentenceBscreenbrBrexBrelationBreflectBraisingBracialBpracticeBportionBparallelBoverlookB
nominationB	nicholsonBmundaneBmiracleBmethodsBmartyBmBlimitsBlengthyBlemmonBjuneBinvasionBhankBfuryBenemiesBembarrassmentBeatenBearsBdownhillBdoubtsBdolphBdinosaurBdimensionalBcureBchessBcalmB	buildingsBborrowedBblowingBbeattyBawakeBarrivedB	alternateBadvertisingBabysmalB1980BwrappedBwillieBvaguelyBtreatsBtimonB
thoughtfulBtaxiBtaughtB	succeededBstevensB	slaughterBsincereBsignedBsidBshowdownBshowcaseBshelleyBsecretlyBroyalBrockyBrisingBrecognitionB
pretendingBpotentiallyBphantomB	operationBnationsBmonkBmiseryB
lacklusterBlBkennedyBkarlBjuniorBjoinedBiraqBiranBincomprehensibleBincompetentBfulciBfrequentBfirstlyB	financialBfacingBeightiesB	educationBedgarBeagerBdonnaB
disappointBdevilsB
definitionBdaltonBconstructionBconfrontationB	conflictsB
complexityB	compelledBcombineB
christiansB	celluloidBbrendaBbathBarmedBappreciationBamericasBairplaneBaidsB2007B	wrestlingBwisdomBwetBwarriorsB	verhoevenB	vengeanceBtrainedBstonesB
stereotypeBshawBsevereBreadsBratsBrabbitB
prostituteBpoetryBplannedBmallBlolBkayBjustinBjennyBintentionallyBheightsBharveyBfooledB
fascinatedBexchangeBempathyBelmBelegantBdreckBdefinedBconcernBconBcivilizationBchannelsBbuttonBblakeBbelaBbareBairportBaccessB1983BwritesBwizardBwishingB	voiceoverBvisitsBusefulB
underneathBtwilightBtroublesBtigerBtagBsueBspiritedBsopranosBsimpsonBshirtBruinsBpressureBpotBplotbrBolivierBnycBnutsBmollyBmightyBmadonnaBmacyBloyalBlowestBjulianBironBinventedBhilariouslyBgermansBfoulBfixBfasterB	fashionedBexceptionallyB	equipmentBeastwoodBcrucialB
creativityBclownBclosedBclicheBchicksB	carradineBbluntBbladeBaustinBangelaBalertBacidBaboutbrB1940sBwoundedBwoundB	upliftingB
suspiciousBsuspendB	survivingBspokeBscotlandBrubyBrottenBrootsBrobinsonB	renditionBremakesB	relativesBrandomlyB
punishmentBpsychicBpolishedBpoeticBpoemBpitBonlineBnonsensicalBmoronicBmerylBmarsB	marketingBloadedBlitBlawsBkyleB	introduceB
helicopterBgregBgreedyBgatherBfirmBexposureBethanBelephantBeitherbrBdutyBdickensBdepictsBdarrenBdanielsBdaBcopiesBconservativeBcomposerB	climacticBchristyBchapterBcakeBblondB1990BwhoopiBwellsBvegasBvaluableBunitBtubeBtribeB
terroristsB	supportedBsufficeBstoogesBstaleBspeciesBsissyBsimultaneouslyBsignificanceBseriesbrB	septemberBschoolsB
richardsonBreceivesBratBquitBpunkBprisonerBpierceBperformsBpeakBopinionsB	nostalgiaB
middleagedBmetaphorB
mentioningBmasterpiecesBlosersBloneBlettersBinteractionsB	integrityBhopperBhistoricallyBheartbreakingB	guaranteeBgarboBeugeneBeducationalBdistinctBdislikedB	dimensionBdementedBdamonB	crocodileBconstructedB
confidenceBchoosesBcapitalBbelongBbeliefsBbehaveBbangB	authorityBattendBassignedBassaultBarrivalBalecBzombiB	wellknownBwarmthBvoicedBsquareBsoundingBsmilingBslickBsinkBsharonBresponsibilityBpromisesB
progressesB	primitiveBpopsB	plausibleBpeoplebrBpalaceBopportunitiesBnicoleBmummyBmistressB	masterfulB	interestsB	inabilityB
homosexualBgraysonB	generatedBfondaBfascinationB
equivalentBeditionBedgyBdrearyBdixonBcravenBcdBbumblingBashleyBantwoneBamitabhB
additionalB3000BworkerBwitchesBwidelyBvisibleBustinovB
unoriginalBtravestyB
transitionB	testamentBtechnicolorBstunnedBsneakBsabrinaBresistBresidentBpursuitBpolanskiBpettyBpartiesBoverlongBnonstopBnewlyB	minutesbrB
mechanicalBmateB
literatureBknockedBjoinsBintimateB
inevitablyB	imitationBillnessBhersBgerardBfunbrBfoughtBfamilysBdominoBdomesticBdistributionBcontestBcoastBclausB	classicalBchoreographedBbridgetBbikoBbernardBballsBbaldwinBawhileB	ambiguousB	alcoholicBactingbrBabusedBwrongbrBwackyBviewingsB	unlikableBunexpectedlyBteachingBtastesBskullBsharedBseeksBrubberBrothBrobberyBrisesBregionBraymondBratesBrangersBrandyBpreviewsBpatienceBpanBpackageBmodeBminimumBmessedBlundgrenBkeithBitemsBisraelBinexplicablyBinappropriateBhydeBhopkinsBharmB	hackneyedBgreedBgenerationsBframedBflairBfidoB	expressedBelB
directionsBdescentBcreatorB
complaintsBchampionshipBcarrieBcarefulBbutlerBbudB	brutalityBboneB	biographyBballetBbacallB
accomplishBteachersBsurvivorB	subjectedB	strangersB	shouldersBsentinelBscroogeBrudeB	resourcesBrequireBreidB	recordingB	prisonersBprestonB
presentingBphonyBpantsBnailB	murderingBmilkBmentionsBmasksBmanipulativeBliBlegalB	laughablyBlandingBkickingBinstinctBhopelessB
frightenedBfeelgoodBfavourB	evidentlyBelsesBdustBdroveBdownbrBdoomBdesertedBderangedB	depictingBdemandBdeceasedBdanaBconneryB	conceivedB	companiesB	comediansBcanyonBbusinessmanBbullBbootB	behaviourBbabeB
audiencebrBattorneyBaspiringBaliciaBadaptationsBwifesBwellwrittenBwakesBviceBvanceBuncutBtownsBtoothBtankBsumsBsubparB	stretchedBstraightforwardBstarkBsolutionBshouldveBsciBroommateBretiredBrestoredB
respectiveBreleasesBrefusedBraveBraisesB	preciselyBpg13BpassableBpamelaBoldestBnerdBnathanBmorbidBmaniacBlenaBlegacyB	immediateBiconicBiconB	heartfeltBhearsBgrabsBgiftedBfunctionBfiBexaminationB
enthusiasmBdysfunctionalBdressesBdianaBdesiredBdawsonBdammeB
compassionBcloseupBcheckedBcannibalBblastBberlinBawfullyBaltmanB	acclaimedBacceptsB1973BwithbrBwhollyBwaitressBvalleyBunpredictableB
underlyingB	trademarkBtrackingBtopnotchBtomatoesBtendsBtackyBswingBstuartBspringerBsoxBsliceBshanghaiBshakyBshadesB	sentimentBsatisfyBreplaceBremadeB
reflectionBrecordsB	receivingBrainesBproceedsBprepareBphotoBpearlBpaulieBpalanceBpalBorderedBoceanB	musiciansBmuseumBmorrisBmannBliteraryBlandsBkathrynBjoshB
indicationBheightBgundamBguestsBglowingB	gentlemanBfedB
expeditionBeditBeasternBearBdrakeBdesiresBdefenseBdaybrBcringeB
conditionsBclipBcitizensBcinemasBchristianityB	capturingBcannonBassumingBamusedBalternativeBaimB	affectionBaccuracyB13thByardB	witnessesB	witnessedBwendyBwalshBunrealBtylerBtrailBsungBstumbledBstudyingBspainBspaceyBsirkBseverelyBromeroB	resultingB	respectedBreferredBreevesBreelBrecycledBprizeB	policemanBpatientsBpartnersBparsonsBobjectsB	objectiveBmtvB
misleadingBmayorBmatchedBmarshallBmapBmadebrBloyaltyBlaurenceB
landscapesBkittyBhelloBguinnessB	grotesqueBfilthBfieldsBexperimentsBexperimentalBestherBemperorBdifficultiesBdeniroBdemonstratesBcrawfordB	correctlyBconfessBcomfortBcharacterbrB
challengesB
challengedBburstBbuffaloBbesideBbayB	barrymoreBastonishingBassumedBassassinBapeBalcoholB
activitiesB1984B1978B1971BwweBwrapB
wildernessBwangBvulgarBunseenBtomorrowB	suggestedBsharesBscreensBscoobyBrussiaBroofBritaB	reluctantBrelaxBrecommendationBrajBquinnBpuppetBpunBpromoteB	profanityBpreposterousB
popularityBpoisonBphillipsBpacificBotherbrBoffbrBnooneBninjaBnephewBnaughtyBmeritsBmelissaBmankindBmaidBkubrickBkissingB
irrelevantB	insuranceBinsipidBimprovementB	greatnessBglobalBfrankensteinBfestBfemalesBelectricBdudBdinBdespairBdatesBcrystalBcostarBcopeBconventionsBcoBboomBbargainBauthorsBanalysisBadoptedB	admirableBabrahamB75B35B1997B1993B1968BwelldoneBvitalBunfairB
translatedBthingbrBstandupBsparkBsoonerBsixtiesBshoddyBseldomB	scarecrowBsandraBresembleBpinkBpaltrowBoutingBomenBollieBmuslimBmusicianBmuddledBmorallyBminusBlouiseBlockBliftedBlesBkansasBjawsB	ignoranceBhughesB
hopelesslyBhaplessBgrahamBglassesBfingersB	exploringBensuesBdisorderBdiscoveringB
difficultyBdelBcostarsBcorpsesBchargedBbrendanBbergmanBbeatlesBantsBalfredBalbumBaddressB
acceptanceBabsentB1933BwildlyBweatherBweaknessBunfoldBtriangleBtowersB
togetherbrBtierneyBthoughtprovokingBteachesBstoresBsorelyB	sillinessBshtBrooneyBrodBrespectivelyBregardedBreferBrealmBpurposesBprovocativeBpropertyBproB	prejudiceBphilB	paramountBownersBnowbrBnicolasBmurrayBmuppetsBmoreoverBmaloneBlushBlocalsBkumarBinsistsB	householdBhealthyBharrisonBgriffithBgregoryBginaBgemsBfrustratingBfreaksBfrancoBfogBfillingBfemmeBfeministB
expositionBexploresBeternalBdesignerBdemiseBcreditedBcrazedBcoupBcomparisonsB
commentingBclanBchoppyBbrosBbilledBbikeBbeverlyBbettyBbasingerBantonBanilBambitionBagendaBaffairsB	absurdityB19thB1987B1986ByetiBwretchedBwouldbeB
widescreenBvibrantB	unrelatedBtokenBtobrBswimB
suspensionB	speciallyBsoftcoreBslimyBshoutingB	selectionBseedBrolledBriotBrelyBreignBrecognizableBpreyBpremiereBparadeB	paintingsBoutfitsBnoisesBmclaglenBmassesBleonBkolchakBinvitedB
insightfulB
imaginableBhulkBhiresBharmlessBhandlingBhammyBguitarBgamblingBfontaineBfillerBfiftyBdesBdeeBdanishBcrueltyBcreepBcreamBcountsB
conscienceBconceptsBchillsBchildsBchampionBchainsawBcatchingBcastsBcassidyBbridgesBbondageBbendB	befriendsBbaconBaudreyBapplaudBabruptB1995B1989ByearoldBwindowsBvisitingBvictoryBventureBtripeBthreadB	terrifiedBteamsBsunnyBstumblesB	strongestBstargateBspinalBsoupBshoppingBsensesB	secondaryBschlockBrukhBrhythmB	repeatingBquaidB	prominentBpostedB
positivelyB
portrayalsBplacebrBpennyBpaxtonB	partiallyBparanoiaBorleansBnodBnewerBnatalieBmolBmillB	mentalityBmegBlumetBkungfuB	knightleyBknightBinhabitantsBinfectedBindependenceBhkBheadingBhangsB
guaranteedB	glamorousBgenieBfuriousBfeedingBenteringBdukesBdoseBdonebrB	deservingBdeletedBcoupledB
convictionBconcludeBcoincidenceBcancerBcampaignB	camcorderBcaliberBboastsBbabiesBawaybrBattractBartworkB
adolescentBaceB
accuratelyB1936BvotesBvotedBvisitedBvisionsBvainB
underworldBtunnelBtriviaBtripleBtrendBsuitablyBsingersB	sincerelyBshortcomingsBservantBscreenwritersBrossBrepresentationBrenderedBrenaissanceBremoveBremainedBrefuseBreflectsBredeemBreadersB	pricelessBpredictBpraisedBpokerBpitifulBphysicsBphraseB
passengersBparadiseBorangeBnemesisBnativesBmirrorsBmichaelsBlindsayBlanceBiqBinvitesBinvestigatingBinfoBinexplicableB
industrialBimmatureBhokeyBheelsBguardsBgabrielBfullerBfortyBfarrellB	fantasiesBexcellentlyBelsebrBelevenBdobrB	disgustedBdiehardB	departureBdeafBcueBcouchBconveysB
controlledB	confidentBcommunicateBclockBcelebrationBbulkBbentBautomaticallyBactiveB2008ByoungestByearsbrBweakerBunsuspectingBunexplainedBundeadBtwinsBtemptedBsymbolicBsweptBstareBsquadBsnakesBsmilesBscrewedBscariestBsassyBrompBrolebrBrerunsBquietlyBpredecessorBpostersBpokemonBpointbrB
phenomenonBowenBnervesBmitchumB	misguidedBmarvelBmarriesBmarcBmalesBlocatedBlifelessBladderBknowbrBklineBinterestinglyBinjuredB	ingeniousBhomesBhippieBhilarityBhelplessBhartleyBgillianBgenderBgearBgateBforcingBfamilybrB	encourageBemployedBdroppingB
disappearsBdevicesBdestinedB	dependingBdelicateBdebateBdamnedBcomplainingBclaustrophobicBcircusBcharmsBcasualBcaricaturesBbiasedBappropriatelyBanywaybrB	amusementBalteredBallensB21stB1988B1979B1945B–ByokaiBwhaleB
weaknessesBwarmingBunderstandsBumaBtrialsBtipB	tastelessBtapesB
suggestionBsuburbanBstaticBstandoutBsniperBskilledBshoulderBrootingBrippingB
restrainedB	residentsBrejectedBrealisedBramonesB
principalsBpgBoffbeatBnorthernBninaBmutantBmuppetBminorityBmindedBmillionaireBlifesBlayersBjoelBivBhomicideBhistoricBheistBhawkeBharoldBgereBgatesBgarnerBfluffBfiftiesBfeatBfadeB	evolutionBethnicBenteredB	endlesslyBemergesBdressingBdowneyB	disguisedBdisguiseB
discussingBdisappointedbrBdidbrB
detectivesBdependsBdazzlingBdamagedBcrackingBconsiderablyBclarkeBclaimingBchickenBcerebralBcenteredBcarlaB	candidateBbuysBbreedBboyleBbowlBbogusBbelushiBattenboroughBasylumBarthouseBafricanamericanB
advertisedBaddictedBabruptlyB73B22B1970BBwellmadeBwaxBwandersBwanderBvoyageBveinB	valentineBusersB	uniformlyBtrafficBtopicsBtokyoBthoughbrBthompsonB
sympathizeB
stunninglyBsteadyBsobrBsleepsBshakingBshakespearesB	satiricalBreportsBrapistBrandolphB
politicianBoutrightBoutdatedBothelloB	organizedBoptionBmutualBmitchBmiamiBloisBlinersBlastsBkeenBjulietBinternalBinsultsBincidentallyBhitlersBherosBheartedBharrietBhandlesBgypoBgoodbyeBgirlfriendsBgimmickBforbrBfistBfishingBfirmlyBfiancéBexploitBexcessB
engrossingBdurationBduckBdriveinBdifferentlyBdetractBdestroysB
describingBdefeatedBdandyBcontributionBconnectionsB	computersB	commanderBchoosingBcharltonBcentsBbunnyBboxerBbowBborisBbordersBbetrayalBbeowulfBbanalBanyonesBactivityB1976BwornBwooBwingsBwigBwatsonBvehiclesBupdatedB
undeniablyB	threatensBtaBsurgeryBstalloneBstairsBstagesBstackBspockBsoughtBsosoBslugsBsketchBsinkingBsiblingsBservingBrouteBroundedBridesBreaderBpursueBpossessBposeyBpeteBpeckBpatriciaBowesBoutlineBobtainBnewmanBneuroticBmontyBmarlonBlimitBjuddBjokerBjillBjarringBinstitutionBinspireBinsightsBinconsistentBhungryBhighwayBhelpfulBgrudgeBgrowthBgalaxyBfrankieBframesBflowersBfiringBfarmerB	exquisiteBentitledBdemonicB
definitiveBdaysbrBculturesB	courtroomB
continuingBcluelessBclashBcheerBcbsBbusterBbtwBbravoB	blatantlyBbittersweetBbatesBbasketB	backwardsBauthoritiesBassassinationBartsyBanyhowBanticipationBakinBaffectsB1994B1969BvocalBvanillaBvaderBunsureB
unlikeableB	ultimatumBtrumanBtowerBtonBthunderbirdsBsupplyBsubtlyBstudiesBspreeBsometimeBsomedayBskitsBsfBserumBsendingBscoopBsalmanBropeBronaldBromeBreviewedBreeveB
rebelliousBpumbaaB
psychopathBpromptlyBprayBpoundsBpeacefulBneatlyBmoralsBministerBmiceBmccoyBmarionBmailBmachoBlovebrBlorettaBlegendsBleapBleadersBlaborB
kidnappingBkidmanB	justifiedBjodieBjewsBjawBhamBhadleyBhackBguineaBgalBfrancesBfoolsB	fairbanksBexteriorBexploitsBexpertlyB
exceptionsBentiretyBemergeBeliteBduvallBdownsBdevastatingB
despicableB
derivativeBdarnBcrossingBcrispBcreekBcrapbrBcompeteBcombinesBclichesBclaraB
chroniclesBcemeteryBcattleBcasinoB
cartoonishBcarmenBboringbrBbittenBbanterBarcBapproachingBalleyBagencyB
accessibleB28B23B1981BByouthfulBwtfBwheresB
werewolvesBwaltBunattractiveB
unansweredBtopsBswallowBstrandedBsteamBspeechesBsomethingbrBsleazeBskitBsimpsonsBsidewalkBservantsBscarfaceBrifleBresolvedBreliableB	realitiesBpunchesBpoleBplayboyBpixarBpinBphotographsBpcB	originalsBnopeBnominationsBnetBnailsBmesmerizingBmercyBmanbrBloweBleighBlaputaBjulesBjaredBjadedBintentionalB	incapableBhippiesBhidesBgungaBgriefBgrainyBgoodingBgeorgesBgableBflashyBflashesB
executivesBeverbrBeventualBendingbrBeinsteinBeatsBearnestBdirtBdepthsBdeliveranceB	delightedBdeliciouslyBcrashingBcontributedBcontinuallyBconanBclimbBcareyBbutcherBburkeBbullyBbroodingB
braveheartBbewareBbestbrBbarrelBauthenticityBarabBantonioBangstBaimingBaidedBadoreB	acceptingBzaneBuniformB	underwearB
underwaterBtossedBthrilledB
terminatorBtaxB	tarantinoBtaraBtabooBsydneyBstringsBstorysBstickingB	startlingBslavesBsixthBshootoutBsharingBseveredBseedyBscrewB	salvationBsaleBriversBrickyBpuppetsBprovingB	preparingBpostwarBpennBpaulaB	parallelsBownedBominousBolBnyBnutBnolteBmythBmumBmouthsBmotiveBmoronsBmixingBminBmayhemBmasterfullyB
mannerismsBluisBlongtimeB
lonelinessBlimitationsBliftBlaurenBlandedBkoreaBjudgmentBjudgedBjerseyBjacketBitemB	isolationBireneB
inaccurateB	housewifeBhindiBheavyhandedBheavensBhandheldBgluedBfrontierBfreezeBfragileBflimsyBfactorsBexorcistB	estrangedBelectionBdudleyBdjBdivineBdistressBdisgraceB
disastrousBdemonstrateBdeclineBcushingBcritiqueB	convictedB
contributeBcontestantsBcontemptB
containingBcommitsBchuckleBchristieBchoppedBchavezBbutchBbuseyBbmoviesBbloomBbeltBbacksBbachelorBbachB	awakeningBassuredBasiaBandreBamritaB	alexandreB95B1977B1974ByoursBxfilesB
winchesterBwardrobeBvastlyBvariedBvanessaBtoniBtobyB
threatenedBtapedBtailBswearingBstrictB	strengthsBstoppingBstalkingBspiceB	spectacleBsgtBsensibleBscriptbrBscoredBromeoB	repulsiveBreminderBrationalBquentinB	publicityB
professionB	preferredBpiratesBpirateBpatternBownsBoverbrBnbcBmoeBmiyazakiBmindbrBmiloBmelvynB
meanderingBmasonBmarthaBlunchBlongingBlastingBlabelBkurosawaBkentBkathyBjediBjacquesBinspirationalBinherentBinformedBimpliedBimoBimmortalBhugelyBgiantsBghettoBgainedBfreakyBfreakingBfordsBfifthBfeastBexposeBexpenseBexpectationBexitBeuropaBerrolBemailBebertBdragonsBcrosbyBcoreyB
commitmentBcomedybrBcharacteristicsBcarellBcapeBbuzzBbrunoB	breakfastBbreadBbostonBbitchBbarsBavoidingB	attendingB	attackingB
astoundingB
approachedBalisonB1991BzorroB
witchcraftBwireBwilderBwhiningBvinceBvileBvegaBtvsBtransformedBtonysB
timberlakeBthirtiesBtheoriesBthelmaBtenantBsymbolB
suggestingBsugarBsubtextBstatingBspiesBshorterBsergeantBselectedBscratchBsandBsafelyBroutinesBrookieBrespectableBregardsB	referringB	redundantBrapidlyBracesBquantumBpursuedBpuertoB	publishedB
psychologyBpromotedBprogramsBproductsBpredictablyBpreachyB	populatedBpfeifferBperformancebrB
perceptionBpeculiarBpaleBpainterB
obligatoryBnormBnieceBnetflixB	mythologyBmotionsBmossBmoronBminsBmarilynBlikeableBlexBletdownBknocksBknightsBkarateBjanesBitselfbrBiranianBintroducingBhybridBhepburnBhayworthBhatefulBhackmanBgroundsBgripB	graveyardBgratefulBgoodsBgodawfulBglobeBgilbertBforgivenBfilthyBfeminineB	fastpacedBexpectsBenthusiasticB	enigmaticBdylanBdisgustBdenyB	deliciousBdegreesB	decidedlyBdameBcriesBcorbettBcopiedBconsistB
confrontedB
comprehendBcoloursBcolinBcohenBcoatBcliffhangerBcleaningBcheaplyB
caricatureBbronsonBbountyBbombsBbiblicalBbeholdBbeforebrBbackbrBaxeB
approachesBadmiredBacquiredBachievesB99B300B1975ByepBwiderB	unusuallyB	translateBtraceB	tormentedBthievesBswedenB	suspectedBsublimeBstylizedBsteeleBstatueBstationsBspitBsophieBsnuffB	sickeningBsettledB	seductiveBseasonedB	screwballBsamanthaBrewardBrescuedBrelentlesslyBrefersBreadilyBrampageBrainyBradicalBqueensB	principleBpoundBposingBposesBpolicyBpleasingBottoBomarB	obstaclesBnuancesBnovakB
noticeableBnarrowBmysticalBmodestBmessingBmermaidBmathieuBliuBliteralBlethalBlendsBlendBlasBkeysBkazanBjulietteB	intricateBinteractBinsanityBimmenseB	horrifiedB	historybrBgoldblumBglaringB	gentlemenBfrogBfreddysB	firstrateB	festivalsB	europeansBeternityB	establishBerrorBenhancedB	energeticBdumpBdrinksB	demandingBdelightfullyBdebbieBdaylewisBcyborgBcounterB	convincesBconsequenceBconradBconfrontB	completedB
collectiveBcloneBcinemabrBcherBcentreB
celebratedBcarnageBbillsBaussieBauditionB	assembledBarrivingBadditionallyBaddictBactorsbrB4thB200B1982B1950B1000ByawnBwhoreB	wellactedB
watchingbrBverdictBuhBturmoilBtunedBtraitsBtorontoBtendencyBswitchedBsurvivesBstewartsB
soderberghBslideBsethBserialsBsammoBsalesmanBroundsBrepliesB	relevanceBrehashBrealisticallyBramblingBprecodeB
possessionBporterBpocketBpayoffBpadBoriginsBodysseyB	occasionsB
noteworthyBnaschyBmisunderstoodBmirandaBmedievalBlongestBlionelB	lightningBlesterB
legitimateBkinnearBkellsBjordanBjennaBjacksonsBinsomniaBinsistB	incorrectB	inclusionB
improbableBhornyBhinesBhermanBhatsBgusBgreatbrBformedBfollowupBflavorB
fassbinderB
farfetchedBexperiencingBexcruciatinglyBevelynBestablishingBenhanceBdumberBdreadB	dominatedBdivorcedBdismalBdefiesBdeerBdahmerBcoveringBconveyedBcollectBcloudsBcliveBclassyB	cigaretteBbyeBbwBbudgetsB	breakdownB	brainlessBbellyBaustenBarrangedB	armstrongBarguingBapplyBadaptionB	absorbingB21B1985B1920sByoutubeBvolumeBvirtualBupsideBupdateBunimaginativeBunBtromaBtrapsBtrainsBtoolB	tolerableBtireBtacticsBswingingBstellaBstalkerBspiderBsondraB	sebastianBscottsB	sarcasticBsaintBrunnerBrevoltB
resemblingB
relentlessB	regularlyBrearBracingBpsychologistBprosBproceedBplanesB
pedestrianBpassageBparticipantsBpaintsB
optimisticBoctoberB	obsessiveBobserveB	nightclubBmuteBmoviemakingBmormonB
misfortuneBmidstBmessyBmeantimeBmealBmccarthyBmannersBmamaBmacabreBlukasBlongoriaBlolaBlighterBliamBleatherBkidnapBjuanBjealousyBinviteBinmatesBindianaBignoringB	identicalB	holocaustBhabitBgrassBgarlandB	fishburneBfataleB	explosiveB	explodingBexplodesBexclusivelyBescapingBensureB	enchantedBegyptianBeducatedBearnBeagerlyBdumpedBdumbestBdocumentBdistractionBdistractBdeputyBdepictBconsiderationB
complimentBcombsBcoffinB	christineBchongBchillBchamberlainB
cassavetesBboyerBbonnieBbikiniBbegsBarrowBarquetteBarkinBarielBaptBanxiousB	antonioniB	anthologyBansweredBalotBagonyBadrianBaccountsB1998B1939BwoundsBworsebrBwiselyBwineBwartimeBvanityBunsympatheticBunsatisfyingBturdBtouristBtongueincheekBtargetsB	suspicionBsuckerBstuffbrB	stephanieBsteerBstanceBstabbedBspanBsmellBslipBsholayBshiftsBsg1B	separatedBsentimentalityBseniorBschemingBscheduleBrupertBreverseBrebelsBrealizationBpushesBproducesBpoppingBplatformBpickupBpalmaB	overblownBoccultBnewcomerBnerveBmurkyB	murderersBminnelliBmatesBmagicianBlunaticBlowkeyB
liveactionBlinkedBlaurieBjewelBislandsB	instinctsBindicateBhypedBhoBheathBheadacheBhavocBgroundbreakingB	furnitureBfrightBfluidBfillsB	extensiveB	exploitedBexcruciatingBentertainingbrBembraceBedithBeconomicBdubiousBdreamyBdramaticallyBdiverseB	dillingerBdilemmaBdiaryBdeterminationB
depictionsBdeniseBdemonstratedBdefineB	deathtrapBdealerBdashingBdarthBdafoeBdadsBcrippledBcreasyB	conductorBchibaBcheekBcharacterizationsBcharacterisationBchamberB
carpentersBcapacityBbsgBbranaghsBbootsBbonesBbleedBbiopicBbeggingBbeckhamBattendedBartistryB
aggressiveB	aftermathB85B1992B1940BwormsB	worldwideBwastesBwarnsBvargasBtrivialBtremendouslyBthrustBtargetedBsyndromeBsubwayBstrokeBsterlingBstardomBspineBsparksB	spaghettiBslimBsketchesBshoreBshepherdBshepardBsensebrBsellsBseemingBseebrBseatsB	scriptingB	scoobydooBrussB	rosemarysBromancesBrightbrBriceB	revolvingBresultedB	repressedB	remindingB	releasingBreiserBrandallBrampantBpsycheB
progressedBpredatorBpotterBpornographyBploddingBplanetsBpenelopeBpansBownbrB	overboardBoriginBoharaBnivenBnarratedBmustveBmobileBmissileB	madeleineBlurkingBlipBlionsBlindyBlaunchBkellysBjudeBjoseBinvestedBintactB	instancesB	incidentsBimpliesBhowlingBhostageBhomosexualityB	homicidalBhardenedBgoodlookingBgeorgiaBgenrebrBgenerateBgeishaBfundingBfrozenBfoxxB
foundationBfortiesB
forgettingBforemostB	followersBflickbrBfleshedBexperiencebrBevansBenvyBencounteredBemployeeBdynamiteBdrabBdominicBdiscoBdisappearanceB	deliriousBdeedBdeathbrBcrownBcrossesBcorporationBcornBcookieBconsequentlyB
compensateB
comparableBcolumbiaBclooneyBclickBclayB	christinaBcarlitosBcannesB	breathingB	blandingsBbikerBberkeleyBbauerB	backstoryBbachchanBarrestBarrayBappliedBappealsB
antagonistBampleBamidstB	ambiguityBallyBadmittedB34B1959B1944ByellBww2BwipedBwhitesBwayansBwagnerBvivianB	victorianBveteransBvalidBuniformsBultraB
traditionsBtonesBthoBtcmBtalkyBsweatBssBsourcesBsourBsopranoBsinsBsightsB	semblanceBseduceBscorseseB	scenariosBsandersBruiningBrowlandsBrivalryBritchieBriderBreuniteB	remembersBrelatingBraunchyBratsoBquestioningBpuzzleBpredecessorsB	possessesBpoliticiansBplottingB
playwrightBpickfordBphonesBparkingBparanoidBpapersBpacksBopenlyB	offscreenBnoveltyBnolanBmysteriouslyB
motorcycleBmickBmeteorBmensBmartianBmadsenBloyBlaterbrBkubricksBkirstenBjulyBjolieBjjBjigsawBjessBjeBiraBinvestigatorB	inventionBingridBhuntedB
hitchcocksBhappenbrBgloriaBglimpsesBgilliamBgiganticBgarfieldBgangsBfriedBfranticBfoolishBfleetBentranceBenduringBelevatorBdividedB
distractedB
denouementBdenisBdecidingBdeannaBcycleB	curiouslyBcubanBcoworkerBcormanBcookingB
convenientBconcentrateBcomebackB
colleaguesBclichedBcirclesBchewBchesBcenaBcelebritiesBcecilBcapBbridesBbratBblamedBblaiseBbenefitsBbabesB
assignmentBartyBanytimeBallegedBajayB	afterwardB	addictionBacknowledgeBabortionB1953BzB
wonderlandB
witnessingBwidowedBwhybrBwakingBvintageB
villainousBverbalBupcomingBunhingedB	unfoldingBtraumaBtomeiBtimmyBticketsB	throughbrBsurroundingsBsubsequentlyBstereotypedBstartersBstarsbrBspencerBsonnyBsmashBsitcomsBshotgunB
shockinglyBsheetsBshaggyBserbianBsearchedBscreenedBsarandonBsamuelBsaltBsailorBroughlyBripsB	rewardingBrewardedBrespectsBrememberingBrejectsB	reflectedBreaBranchBquarterBqualifyBpuppyB	premingerBpointingBpivotalBphiloBphantasmBpauseBparksBpainsBpB
outlandishBorphanBordealBoliviaBoldsBobservationsBnuancedBneedingB
moviegoersB	momentsbrBmockBmiraculouslyBmiikesBmcqueenBmanicBlucilleBlopezBlavishBlastlyBlabeledBkinskiBjuryBjanetBindifferentBhomebrB	harrowingBgretchenB	gatheringBforgivenessBfoilBflowerBflockBfilmographyBfanaticBenoughbrBembarrassinglyBeliBduelBduBdougB	diversityBdismissBdevotionBdespiseBdemilleBdeemedBcurlyBcrushedBcrossedB	criticizeBcrashesBcowBcountyBcountingBconventBconvenientlyBconsBconfuseBcommendableB	colleagueBclerkBchucklesBchoirBcheersBcharliesB	centuriesBcaronBcampusBcampingBbreastBbrashearBbossesBbertBbegBbatsBbakshisBbackedBavoidsBavidBassureBarebrBaboundBabandonB19B101BwrightBwhinyBwhereverBweeklyBvoidBupbeatB	unnaturalB	unleashedBuncannyBumBughBtrierBtoolsBtherapyBsunriseBsundanceBsummedBstardustBstalkedBspoilingBslappedB	sincerityB
similarityB	silvermanBshrekBshapedBseymourB	sentencesBselfindulgentB	schneiderB
scarecrowsB
sacrificesBruledB	rodriguezB	robertsonBrivalsB
resistanceBreplacementBrepeatsBragingBposeBpollyB
phenomenalBphaseB	pervertedBpazBpattyBpathosBoverbearingBothersbrBoldfashionedBoffendBofbrBnorthamBnetworksBmovieiBmortalBmomsBmeyerBmentorB
melancholyBmaximumB	mastersonBmaskedB	marijuanaB	maintainsBlicenseBlayingB
laboratoryBkeatonsBjudgesBjarBinterruptedB	intenselyBinsertB
incidentalB
incestuousBinadvertentlyB	immigrantB	imaginaryBidolBhystericallyBhopBhootB	himselfbrBharronBgutBguardianBgrabbedBgoalsBgielgudBgalleryBgadgetsBgacktBfundamentalBfuelBfratBfiresBfeebleB
favouritesBfadesBfactualBexcusesBenBelsaBelliottBeffortlesslyBealingBdustinBduhBdooBdivisionBdistrictB
displayingBdestinationBdeskB	defendingBdebtBdaisyBcypherBcutterB
cunninghamB
conventionBcontroversyB	considersBconfinedBcolmanBcollinsBcolletteBclaudeBchloeB	childlikeBcheadleBcastbrBcarpetBcapoteBbugBbrookeBboutBbernsenBbcBbattlingBbarbraBawfulbrB
apocalypseBangelinaBangB	ambitionsBachievementsB
accidentalB1948B10brBwrestlerBwongBwinnersBwheelBwendigoBwarrantBwarholsBvoyagerB	vignettesBuneasyBunderstandablyBunappealingBtoxicBthurmanBtakerB	surrenderBsupremeB
sufficientBsuckingB
substituteBsubstantialBstumbleB	spoilerbrBsoylentBskinnyB	skepticalBsheepBsheenBshahidBservicesBsergioBsensibilityBsatisfactionBsanityBrosesBroboticB	reviewingBresumeBreservedBrepublicBreportedBreeseBredneckBrecreateBrecklessBreBqBpythonB
protectionBpreciseB
possiblebrB	picturebrBpiaBpathsBparticipateBparrotBpalmBoutrageouslyB	orchestraBobservationB	obscurityB
nauseatingBmusicbrBmotelBmonksBmelodyBmanipulatedBmaldenBmaintainingB	macmurrayBleanBkruegerBkatieBjokingB	johanssonBjoBjanBjacksBiturbiBintroBinteriorBinjuryBignoresBhustonBhayesBhandicappedBgwynethBgratingBgrandmaBgracesBgorillaBglanceBgigBfuzzyBfruitBfrontalBfrombrBfranklinBflowsBflamesBflagBfinishesBfarmersB	fairytaleBfacilityBexplodeBeverettBepitomeB
enormouslyB
enchantingBechoesBearliestBdwightBdrawingsBdraggingBdirecttovideoBdallasBcunningBcradleBcommunicationB
committingBcoloredBcoburnBclunkyBclosureBclarityBcindyBcheerfulBcheechB
captivatedBcanceledBbuttonsB
boundariesBbonBbogartB	bodyguardBblockbustersBblamesBbitingBbiasBbennettBbenjaminB
beckinsaleBbaddiesBanticipatedBamazonBalaBaffleckBadmitsB	admissionBaccompanyingBabuB
youngstersByellsBwoefullyBwaynesBwardenBvividlyBvincenzoB	viewpointBvergeBunstableBuninspiringBunderdevelopedBturkishB	traumaticBtransparentBtombBtitularBtideBthunderB	throwawayBthailandBtemperBtakashiBsuzanneBsustainBstudiedBstreamBstilesBstakeBstadiumB	spidermanBsolvedBslyBsinksBshredBshocksBshiftBshannonBshahBsessionBsecureBscriptwriterBscarlettBsatanicBsarcasmBsaloonBsailorsBrourkeBrollsBrealisesB
protectiveB	preachingBpoeBpennedBpartbrBpalsB
originalbrBmyrnaB
monotonousB	monologueBmoneybrBmedicineBmeadowsBmarcelBlynchsBlistenedB	lingeringBkareenaBjustificationBjobbrBinterviewedB	interplayBinterestingbrBinformativeBinaccuraciesBhumBhostelBhawnBhallmarkBhainesBgrooveBgoldieBglowBgarageBfraudBfiendBfanningBfacebrBexistingBexcelsB	espionageBernestBentriesBengineB	employeesBedisonBdynamicsBdrowningBdriversBdistributedBdistinctionBdisappointsBdisabledBdiegoBdeadpanBdamselB
criticizedBcontrollingB
contestantBconstraintsB
commandingB	combiningBcomaBcmonBcloakBclimbingB
classmatesBcheeringBcandleB	cancelledBcainBburntBbuddingBbubbleBborrowsB	bloodbathBblessBblackandwhiteBavengeB	automaticB	associateB	arroganceBanchorsBalonebrBaircraftB	affectingBadaptBabominationBaaronB010BzodiacBwesleyBwatcherBwashedB	vigilanteBvetBvalBusbrBunravelB
unfamiliarB
transformsBtormentBtonedBtitsBtherebyBtheatresBtestingBtaimeBsubgenreBstrungBstraighttovideoB
stepmotherB
statementsB	spotlightB
somethingsBsociallyBsmugBslowerBshoeBshawnBsensitivityB	sasquatchBsangBsaneBsalvageBrussiansBromaniaBrodneyBrobbersBrobbedBrightlyBrewriteBreunitedB	renderingBremarkBrelaxedB	rebellionBramBraidBquintessentialB
principlesBpresidentialB	practicalBpoppedBplateBpizzaBpilotsBpigsBphoenixBphillipBperverseB	permanentBperilBpeckerBorganizationBoprahBobrienBninjasB	neglectedB	necessityBnataliBmooresBmaureenBmarisaBmanipulationB	magazinesBlockeBlinearBlilBlensBlandmarkBkrisBkindlyBjuiceBjaggerB
investmentBinterpretationsBintendBintellectuallyBinformsB
imprisonedB
idealisticBhmmB
happeningsBhannahB	griffithsBglossyBgloomyBgiovannaBghostlyBghastlyBgesturesBgapsBgapBfritzBforrestBforgetsB	finishingBfascistBfairnessBexploitativeBevokesBevokeB	elephantsBedmundBdunneBdrownedBdreyfussBdominateBdivingB	distortedBdistinctiveB
disciplineBdimB
deliberateBdebraBdaytimeBdavidsBcryptBcrashedBconnieBconclusionsB	collectorBcollaborationBcokeBchopsBchipBcasperB
casablancaBcarlosBcapricaB	cameramanBbuffyBbrentBboobsBboardsBblondellBbeanB
battlestarBbarnesBbanksBbalancedBbadnessBaztecB	awarenessB
astronautsBassetBaroundbrBappliesB	annoyanceBannoyB	animatorsBangieBambianceBaltmansBalmightyB	alexandraB	aestheticB1967B1963BwhodBwearyB	warehouseBvomitBvibeBvcrB	variationBtinaBtightlyB
thereafterBtendedB
temptationBtechBtashanBsystemsB	switchingBsunkBstupidbrBsteamyBsteamingBstabBspontaneousB	spaceshipBsophiaBsnowmanBsnipesBslaveryBslashersBsimplerBsandyBsaintsBromanianBrocketsBrevivalBreluctantlyBrejectB	reasoningBratioBpsychedelicBprotestBprologueBprogressionB	programmeBpretendsB
prejudicesBpornographicB	placementBpierreBperspectivesBperiodsBpassionsBpairingB	overtonesB
ostensiblyBoscarwinningBongoingBnoticesBnorrisBnightmarishB	motivatedBmonroeB	miyazakisBmistyBmisterBmidgetB
mercifullyBmarvinBmarineBmangaBloonyBliottaBlifelongBlibertyBkristoffersonBjoiningBisabelleBinsertedBinformerBinexperiencedBinclinedBimhoBillusionBhookerBhiphopBgreaseBgovindaB	gladiatorBgentlyBgeneticB	functionsBfileBfightersBfederalBfashionsBfartBfamedBewoksBepicsBensueBdriftBdownbeatB	disregardB	discussedBdinerBdernBdecencyBdashBcursedBcringingBcravensBcortezBcontrolsB	consciousB	confrontsBcompositionBcomplicationsBclutterBclinicBclientBcladBcircaBcharacteristicBchaoticBcedricBcassieB	caretakerB	butcheredBbustBblessedBbegunBbeardBbashingBarmorB	argumentsBappleBapolloBallisonBahmadBadmirerB	abundanceB26B1966B1951BzizekBzanyByearbrB	wholesomeB	whimsicalBwarnersBwalmartBvoodooBvisitorB	versionbrBversaBursulaBunnecessarilyBunconventionalBunclearBtruthsBtroopersBtravisB	transportB
transplantBtautBsurroundBsurfB
supportiveBsupermarketB	submarineBstrainedBspoofsBspiralBspellingBspelledBsophisticationBsmithsBslippedBsleptBslaterBsicknessBshelvesBshaolinB	shamelessBseussBsensationalB	sensationB	seductionBseBscreenplaysBsammyBsalesBrousingBrolesbrBrkoBritualBridersB
retirementB	retellingBresolveBrenoBrelyingBrealbrBrapidB	radiationB	qualifiesBquaintBproudlyBprostitutionBpropB	projectedBprofessionalsBprefersBpompousBpolishBplugBplotlineBplacingB
paperhouseBpairedBoutlawB
oppositionB
officiallyBoccupiedBnoamBniftyBnearestBnatashaBmyraBmurielBmostelBmobsterBmishmashB
mediocrityBmarkedBmarinesBltB	lookalikeBlonerBlocateBliftsBlevyBlegionBkidnapsBkennyBkeanuBkathleenBkBinformBinfluentialB
infinitelyBincreaseBilB
identitiesB
highschoolBherbertBhenchmanBheavenlyBhazzardBhauntBharilalBhansBgruffBgrosslyBgrislyBgraduateBgolfBgiBgeekBgatheredBfreedBflashingBfieryBfierceBfiancéeBfetishBeyesbrBernieBerBelevateB
electronicBelectedBdvdbrBdunstBdukakisBdrummerBdreamingBdoyleBdodgyBdiazBdemeanorBdeedsBdarklyBdangerouslyBczechB	customersBcrooksB	coworkersBcowardlyBcourtesyB
courageousBcorkyB	conveyingB	connivingB	condemnedB
complainedBcolonyBchopBchargesB
censorshipBcaperBcabBbusbyB
boyfriendsBborrowBberengerBbelleBbatwomanBbattlefieldBaweighBawaitingBaugustBatrocityBatlanticB
assistanceBarcherBapplauseB	announcedBanitaBanalyzeBadvisedB	addressedBaccomplishmentB51B1958B1943B1932BzuByBwrathBwolvesBwhipB	weirdnessBwashBvertigoB
verhoevensBvaughnBupstairsB
unemployedB	underusedB
undercoverBtrickedBtraveledBtolerateBtodaybrBtepidBtackleBsymbolsBsweepingBsurgeonB	superstarBsunsetB	stumblingBstuffedBstagingBspottedBsnapB
slowmotionB	showcasesB	shootoutsBshoBshieldBsharksBshamefulBsenatorBsemataryBsealBscarierBsaraBroadsBrespondsBrespondBrepresentingB
reportedlyB	rehearsalBredgraveBrapesBpursuingB
protectingB	promotingBpriestsBprankBpotentBpolarBpokesBplayfulBpenBpeggBovershadowedBoutsetBoperateB	officialsBoffenseB	obliviousBnutshellBnunsBnotwithstandingBmorseBmondayBmomentumBmoleBminesBmiddleclassB	melbourneBmartiansBmarryingBmarcusBmaguireBmackBluzhinBluxuryBlureBloganBlinebrBlindseyBlikebrBlesbiansBlaserBlarsBlaraBlamasBknockingBkneesB	judgementBjointBirvingBirresistibleBintelligentlyBincestBimplyB	imitatingBideabrBiconsBhurryBhunkyBhugoBhostileBhooperBhometownBhintedBherdBhelmetBheatherBhartleysBhairyBgrinBgreeneBgravityBglendaBginoBgiggleBgeoffreyBgameraBfurBfrostB	fortunateB	footstepsB
flamboyantBfinneyBfenceBfelliniBfailuresBfabricBexplanationsBexpandB
excellenceBenglundB	emphasizeBeagleB	dreamlikeBdrainBdetroitBdependBdenyingBdeckBdealersBdaylightBdaniBcrowdedBcrookedBcracksBcountrysBcounterpartsBcountedBconvictBconnorBcockneyBcocaineBclimbsBchokeB
cheesinessBceilingB	cannibalsBcalBbyronBbtkBbrushBbritneyBbreatheBbreakthroughBbrandonBboostBbitesBbitchyBbeersBbartonB	bartenderBballroomBawardedBassumesB	architectB
apprenticeBantiheroBantiBanguishBamidBallanBaliB
alcoholismBalarmBadvancesB
admirationBabroadB48B1946BzelahBzealandBwronglyBwitsBwillemBwieldingBveraBvaryingBurgencyB	unnervingBunderstatementBtrampBtorchBthugBtastyBtasksBsylviaBsurfersBsupportsBsuperfluousB
successionBsuaveBstirringBsternBstereotypingBstefanBstarshipB
standpointBstalksBspadeBsoberBsleuthBsleepyBsixteenBsheridanBshadyBshadowyBsessionsBsensualBselfcenteredBsectionsBseattleB
scratchingBsatisfactoryBrunawayBruddBromerosBrogueBrobbingBrichlyBrevelationsB	restraintBrepresentativeB	remainderBrelatesBregimeBrazorBprolificBpresumeB	practicesB	policemenB
photographBperceiveBparodiesBoweBoverwroughtBoutsBoutbreakBobservedBnuttyBnovelistBnormaBneverendingBnailedBmjBmixesBmissionsBmischievousBminingBmindsetBmechanicBmauriceB
mastermindBmarquisBmandyBmaidenB	magicallyBluridBlungBlucioBlorreBloBlistingBldsBlatinoB	languagesBkamalBjurassicBjohnsBjewelryB	irritatedB	intellectBinhabitB	impendingBidiocyBicyB	hungarianBhoganBheroinesBhenchmenB	heartlessB
hardboiledBhamillBhallucinationsBgrendelBgrandparentsBgoodmanBgodardBgobrBgibsonBgershwinB
gandolfiniB	galacticaB
fulfillingBframingBfodderBflowingBfetchedBferrellBfearedBeyedBexposingBexamineB
entertainsBengineerB
encouragedBegyptBedgesBearnsBdorisBdishBdisdainBdiscernibleBdiggingB	determineBderivedBdefyBdeepestB
dedicationBcustomsBcrocBcowroteBcowardBcoppolaBcoopBcoolestBcontributesBcontinuouslyBconsumedBconsciousnessBconfinesBcomebrBclothBclawBcinematographicBchewingBchevyBcheungBcaptiveB
capitalizeBburialBburdenBbrilliantbrBbothersBbookbrBbleedingBblazingBbingBbenoitBbehavingB
beforehandBbearingBautoBauraBarticleBarnieB
anythingbrBalliesBallianceBalienateBadsBactionbrBacclaimBabsorbedB700B400B1955B1934ByarnBwokeBwidowerBwhippedBwheelsB
wheelchairBwarbrBwaltersB	violentlyB	victoriasB
upbringingBunconsciousB	uncertainBtuckerBtruthfulBtriggerBtreatingBtransitionsBtouristsBthumbBthroneBthinnerBthingsbrBthatllBtablesB	superiorsB	subtitledBsubmitB
structuredBstrippedBstrainBstabsBspringsBspittingB
spielbergsBsnowyBsnatchBslipsBslaughteredBslamBskippingBsinatrasBsimonsBsimbaB	signatureBsiblingBshelterBshaunBsexistB	sentencedBsemiBsciencefictionBsaddestBruggedBrollerBroachBrhymeBregisterBrebeccaBreasonbrBraptureBpuzzledBpubBpsychiatricBprostitutesBproportionsBprophecyB	prolongedBprogrammingBpouringB
portugueseBpondBplaguedBpitaBpenaltyBpeggyBpeersBpayneBpaycheckB	patrioticB	passengerBoveractsBnunBnotionsBnoraBnestBnapoleonBmythicalBmuslimsBmudBmoldBmiraBmillsBmeredithBmelindaBmeaningsBmatchingBmartinoB	mandatoryBmaeB	macdonaldBmaBlubitschBlombardBlizardBlightweightB	libertiesBledgerBlaysBlawyersBlaunchedBlargestBkaufmanB
kaliforniaBjedBjamBjackmanB
intestinesBinterspersedB	interiorsBinconsistenciesBimpressionsB
impeccableBimitateBillustratedB
illustrateBhypnoticBhopefulBhoneyB	hendersonBheirBhealingBhatingBharlemB
hammerheadBhahaBgovernmentsBgoersBgimmicksBgiftsBgenaBgenBfunkyBfrenzyBfloorsBflipBflewBfayeBfastforwardBfadedBextraordinarilyB
expressingBexpertsBerikaBentertainerB
encouragesB	embarrassBelderBehBeditsBeconomyBechoBdownfallBdopeyBdominickBdomB
directorbrBdillonB
diabolicalBdevelopmentsBdetachedB	depardieuBdefiningBdcBdataBdarlingB	cowrittenBcoverageBcorrectnessB	convertedB	continentB
connectingBconclusionbrBconcentratesBcompoundB
companionsBcohesiveBcockyBclydeBclimateB	civiliansBchimneyBchenBchairmanBcellsB	celebrateBcathyBcastroBcarlitoB
capitalismBbrennanB	brazilianBbondsBblissBbelievablebrBbelievabilityBbehavesBbartBbarneyBatticB	attendantBastonishinglyBashB	artemisiaB	apologizeBanxietyB
animationsB	allegedlyBagobrB	achievingBacademicBaboardB20sB2009BworryingBwondrousBwolfmanBwinstonBwhoveBweaverBvulnerabilityBvotingBvisceralB	virginityBviggoBvetsB
variationsB
undertakerBunbornBtrustedBtraumatizedBtossBtilBthroatsBthreadsBthereofBthemselvesbrBtextureB	terrorismBtediumBteddyBtangoBsykesBswitchesBswatB	surpassesB	supremacyB	stupidestB
strikinglyB	strangestBstonedBstatureBstaresBspellsBslugBsitesBshrinkBshrillBsherlockBseriousnessBscreamedB	scatteredBsaifBsackBrustyBrussoBrosarioBretrieveBrestlessBregretsB
recoveringBrantBpurseBprophetB
profoundlyBprofitBpricesBpreteenBpressedBpremisesB	premieredB	predictedBpranksBportmanBpoppinsBplantsBpistolB	perceivedBpenguinBpatchBpaddedBowningBowlBovertBouttaB	opponentsBoperasBogreBnovemberBnotingBninetyBnerdsBmustacheBmpaaB	monasteryB	moderndayBmmB	misplacedBmeltBmasseyB
manipulateBmanagingBlocalesBlansburyBknackBkibbutzBjoviBjerkyBjanitorBit´sBisraeliBinsistedB	injusticeB	inhabitedBhugBhousebrBhollandB
hobgoblinsBhispanicBhiltonBharmonyBharlowBharborBhalfhourBgustoBgrievingBgretaBgrandpaBgradyBgoshBgeniusesBgeeksBgeeBgardensBgamebrBgageBfussBfulcisBfugitiveBfryeBflorianeBfleeingBflameBfixedB
fitzgeraldBfewerB	fathersonBfarrahBexudesBevolvedB	episodebrBenlightenedB
engagementBelmerBellaB	efficientBecstasyBdutiesBduryeaBduffBdrumBdreamedB
dreadfullyBdrasticallyBdiverBdisappearingB
deservedlyBdelveBdeliaBdeathstalkerBdaresBdaisiesBcurtainBcrowdsB
criticismsBcrennaB	creationsB	confirmedB
conceptionBcompassionateBcollapseBcolemanBcoalBchandlerBcautionB
categoriesBcarolineBcaptBcapsuleB
candidatesBcampersBcallahanBburstsBburgessBbulliesBbsBbrettBbranchB	botheringBbondingBbetraysBbetrayedBbethBbelieverBbearableBbarmanBbaldBbaitB	backdropsBauteurBauntsB	astronautBartisticallyB
aristocratBarchitectureB
antichristBamuseBamoralBalvinBacquireBabstractB98B65B510B
yourselvesB
yourselfbrByearningBwwiBwuBwizardsB	willinglyBwelcomedBwbBwarpedBwaitsBvivahB	villagersBviennaBvicB
vaudevilleBvapidB	vanishingBvanishesBuptightB	unwillingBuntrueBunnamedBturtleBtsuiBtruebrBtransferredBtomatoB	tigerlandBthroughoutbrBthereinBthankfulBtessBtentB	temporaryBtellyBswordsBswitzerlandBswissB	stylisticB	stretchesBstreakBstrayBstinkBstillsBstealthB	spreadingBsportingBspectrumBsoreBsnippetsBsmittenBsloaneBslewBsignificantlyBshowtimeBshoutsBshoutBshinyBshinaeBsheetaBshakesBseriouslybrBsequelbrBsealedBscriptwritersB
sacrificedB
sabretoothBrollercoasterBretainBreplayBreincarnationBrefugeBredfordBrecoverBrecipeB	ravishingBpuriBpsychologicallyB
proverbialB
pronouncedBprofileBproductionbrB	prevalentB
preferablyBpredictabilityBppvBpourBposhBpokeB	pleasuresBplatoonBpetersonBpertweeBpausesBpaulsBpattonB	paragraphBpalmasBoverwhelmedBoverusedBoverseasBoutrageBorlandoBorgyBoptimismBonenoteBomBolympiaBolsenBnerdyBneoBmuscularBmurphysBmuniBmidwayB
metropolisBmessbrBmerryBmcadamsBmarredB
marginallyBmacbethBmabelBlordsBlordiBloopBlongbrBlocaleBlivesbrBlimbsB
lieutenantBleopoldBleapsBlawnBlanzaBlampoonBkordaBkiddieBkeitelBjaffarBitiBitaliansBirresponsibleBindulgeB
immigrantsBimmersedBidealsBhurtingBhoustonBhockeyBhickockBhesitateBhesheBhandyBguruBgroundedBgripsBgomezBgoldsworthyBgirlbrBgaspBfreelyBfreakedBforumB
forebodingBflaviaBflamencoBfauxBfarewellBfanaticsBfableB	expressesBexgirlfriendBestablishmentBerabrBequalsBenthrallingBemmyBelishaBelectricityBeggBebayB	eastwoodsBdwarfBdodgeB
distinctlyBdiscussionsBdictatorBdenverBdellaBdeclareBdeborahBdangersBdaneBdakotaBcrassBcowriterBcorbinB
continuousB	consistedB
confessionBcondescendingBconcentrationB	comprisedB	competingB	communismB	comicbookBcodyBclubsB	classroomB	classicbrBclarenceBcheatBcheaperBchaptersBceremonyBcensorsBcbcBcavesBcatastropheBcasebrBcarelessBcareerbrBcapshawBbritsBbrickB	braindeadB
borderlineBboothBbookerBbombingB	blackmailBberserkBbeefBbastardBbashBbarkerB	backwoodsB	awfulnessB
atrocitiesBathleticBateB
assortmentBassistB	arbitraryB
annoyinglyB	announcesBanistonBalternatelyBalliedBairingBaimsB	admirablyB	addressesBacknowledgedB1957B1942B1931B18thB150BwovenBwisecrackingBweavesBvotersBvirginsBvengefulBunluckyBtrickyBtrendyBtremorsB	transformBtoppedB	toleranceBtobeBtestedB
terriblebrBtensionsBtackedB	suspendedB
supportersBsubduedBstoicBstimulatingBstarvingBstabbingBsqueezeBspadesBsoullessBsorrowB	societiesBsnappyBsmarterBsleeperBsilkBsiegeBshueB	shatteredBshakespeareanBshackBsensibilitiesBsearchesBscrewingBschemesBscarybrBscandalBsammiBrumorsBrottingB	roommatesBromanoBripoffsBrevolverB
retrospectB	replacingB	repellentBrenderBrelaxingB	rejectionBregalBredeemedB	recurringB	recreatedB
recognisedBrecallsB
reanimatorBrackB	punchlineBproneBprimalB	preventedB	pressuresB	preservedBpowsBpostmanB	positionsBpeaksBpatternsBpaddingB	operatingBolympicBoddballBobsceneBnellBneglectBmutedBmutantsB	mortensenB
monologuesBmobstersB
mismatchedBmiraclesBminionsB
millenniumBmeyersB	mcdermottBmarxB
marvellousBmanufacturedBlumpBlotrBlogoBlizBlanaBlampoonsBlambsB	kornbluthBkeyboardBkerryBjoannaBirisBinventBintruderBinsultedB	insteadbrBinnuendoB	inheritedB
increasingBincompetenceBinchBimdbsBicebergB
historiansBhisherBhilaryBhilariousbrBherringsBheroismBhelpbrBhelenaBheiressBheapBharrysBhanzoBhanBhaltBhaleBgypsyBgrumpyBgleeBgainingBfuturebrBfriendshipsB	friendsbrB	frameworkBfortBforsytheBformalBfollowbrBfloatBflemingBfleetingB	firsttimeBfiascoBfernandoBfeaturelengthB
farnsworthBfaintBexpandedBexhibitB	excrementBexaminedBeverythingbrBepisodicBensuingBemploysB
eliminatedBedwardsBdwBdustyBduchovnyBdrumsBdruggedBdrippingBdrainedB
documentedBdistinguishedBdigsBdiamondsBdeviousBdevganBdenialBdefenceBdarkestBcurrieBcontemplateBconnollyB
conflictedB	concludesBconcentratedBcommentariesBclumsilyBchristensenBchoreBchockBchemicalBcelebratingBcasuallyBcarusoBcarnivalBcanonB
calculatedBcafeBbuildupBbryanBbrunetteBbrandosBbombedBbogdanovichBbimboBbehalfBbaseketballBballoonBazumiBawryBautobiographyB
attributesBassociationBarizonaBapprovalBappalledB	apartheidBannoysB	andersonsBandbrBamokBalterBaielloBadventurousB
addressingB86B27B250B1965BzoomBzeniaBwryBwrestlemaniaBworshipBwinsletBwillyBwelchB
voiceoversBvisitorsBviscontiBvirtuesBvinnieB	versatileBvacuousBusageB	upsettingBunwittinglyB
untalentedBuniversallyBtwobrBtroyBtripsBtomsB	tomlinsonBtintinBthornBtheodoreBtheirsBthebrBtextbookB
tendernessBtenantsB
tearjerkerBtanksBsybilBswampB	surpassedB
stretchingB
stepfatherBspectacularlyBspearsB	smalltownBsmackB	skywalkerBshovedB	shootingsBshearerBseventhB
separationBseasideBschtickBscaringBsansBrunofthemillBrigidB
rightfullyBriddenBricoBrewatchBrestoreB
registeredB
recreationBrecommendingBranmaBrangingBpursuesB	prototypeBpremierBposedBportionsBpoliteBpokingB	poignancyBpoetBplaybrBplantedBpicnicBphiladelphiaB
perversionBperiodbrBpercentBpbsBparticipatingBpartialBparsifalBovertlyB	overnightBoveractBouttakesBorientedB
orchestralBopusBoneillBoblivionB	norwegianB	northwestBnightbrBneesonBnasaBmythsBmotifB
moonstruckBmontagesB	monstrousB
moderatelyBmockeryBmichelBmeekBmcgavinBmcBmaturityBmarathonBmanosB
managementBmadmanBmacBluthorBlotbrBlorenzoBlistsBlinesbrBlimpBliliBlightlyBlessbrBlentBlennonBlangeB	labyrinthBkeeperBkeelerBjuicyBjoyousBjoshuaBjockBjanuaryBjamesonBitbutBirsB
irrationalBinterpretedB	interpretBintendsBintegralBinsecureB
influencesBinducingBindifferenceB
improvisedBillustratesB
illiterateBilkB
identifiedBhousingBhornBhorizonBhooverBhoodsB	honorableB
hitchhikerB	historianBhilliardB
hesitationBheritageBhereinBhectorBharpB	hardshipsBhardestBhappierB	hamiltonsBhagenBhabitsBgramsBgossipBgooBgamutBfulfillBfoxesBfostersBforbesBflippingBfleaBfinnishBfiguringB
featuretteBfallonBextremesB
expressiveBevolveBethicsBeppsB	eponymousBenhancesB	empathizeBelliotBeggsBdrippedBdownwardBdownsideBdogmaB	documentsBdistributorsB	disastersBdirectorwriterBdirectionbrBdifferBdestructiveBdesolateBdenseBdeniedBdemonstrationBdemographicBdecemberBdancedBcynicismBcuesBcroweBcrawlBcrackerBcousinsB	corridorsBcopingBconvictsB
consistingBconroyBconquestBconnectsB
compromise
??
Const_5Const*
_output_shapes	
:?N*
dtype0	*??
value??B??	?N"??                                                 	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?                                                              	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?       	      	      	      	      	      	      	      	      	      		      
	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	      	       	      !	      "	      #	      $	      %	      &	      '	      (	      )	      *	      +	      ,	      -	      .	      /	      0	      1	      2	      3	      4	      5	      6	      7	      8	      9	      :	      ;	      <	      =	      >	      ?	      @	      A	      B	      C	      D	      E	      F	      G	      H	      I	      J	      K	      L	      M	      N	      O	      P	      Q	      R	      S	      T	      U	      V	      W	      X	      Y	      Z	      [	      \	      ]	      ^	      _	      `	      a	      b	      c	      d	      e	      f	      g	      h	      i	      j	      k	      l	      m	      n	      o	      p	      q	      r	      s	      t	      u	      v	      w	      x	      y	      z	      {	      |	      }	      ~	      	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	      ?	       
      
      
      
      
      
      
      
      
      	
      

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
       
      !
      "
      #
      $
      %
      &
      '
      (
      )
      *
      +
      ,
      -
      .
      /
      0
      1
      2
      3
      4
      5
      6
      7
      8
      9
      :
      ;
      <
      =
      >
      ?
      @
      A
      B
      C
      D
      E
      F
      G
      H
      I
      J
      K
      L
      M
      N
      O
      P
      Q
      R
      S
      T
      U
      V
      W
      X
      Y
      Z
      [
      \
      ]
      ^
      _
      `
      a
      b
      c
      d
      e
      f
      g
      h
      i
      j
      k
      l
      m
      n
      o
      p
      q
      r
      s
      t
      u
      v
      w
      x
      y
      z
      {
      |
      }
      ~
      
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
      ?
                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                             	      
                                                                                                                                           !      "      #      $      %      &      '      (      )      *      +      ,      -      .      /      0      1      2      3      4      5      6      7      8      9      :      ;      <      =      >      ?      @      A      B      C      D      E      F      G      H      I      J      K      L      M      N      O      P      Q      R      S      T      U      V      W      X      Y      Z      [      \      ]      ^      _      `      a      b      c      d      e      f      g      h      i      j      k      l      m      n      o      p      q      r      s      t      u      v      w      x      y      z      {      |      }      ~            ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?      ?                                                                      	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?       ?        !      !      !      !      !      !      !      !      !      	!      
!      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !      !       !      !!      "!      #!      $!      %!      &!      '!      (!      )!      *!      +!      ,!      -!      .!      /!      0!      1!      2!      3!      4!      5!      6!      7!      8!      9!      :!      ;!      <!      =!      >!      ?!      @!      A!      B!      C!      D!      E!      F!      G!      H!      I!      J!      K!      L!      M!      N!      O!      P!      Q!      R!      S!      T!      U!      V!      W!      X!      Y!      Z!      [!      \!      ]!      ^!      _!      `!      a!      b!      c!      d!      e!      f!      g!      h!      i!      j!      k!      l!      m!      n!      o!      p!      q!      r!      s!      t!      u!      v!      w!      x!      y!      z!      {!      |!      }!      ~!      !      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!      ?!       "      "      "      "      "      "      "      "      "      	"      
"      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "      "       "      !"      ""      #"      $"      %"      &"      '"      ("      )"      *"      +"      ,"      -"      ."      /"      0"      1"      2"      3"      4"      5"      6"      7"      8"      9"      :"      ;"      <"      ="      >"      ?"      @"      A"      B"      C"      D"      E"      F"      G"      H"      I"      J"      K"      L"      M"      N"      O"      P"      Q"      R"      S"      T"      U"      V"      W"      X"      Y"      Z"      ["      \"      ]"      ^"      _"      `"      a"      b"      c"      d"      e"      f"      g"      h"      i"      j"      k"      l"      m"      n"      o"      p"      q"      r"      s"      t"      u"      v"      w"      x"      y"      z"      {"      |"      }"      ~"      "      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"      ?"       #      #      #      #      #      #      #      #      #      	#      
#      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #      #       #      !#      "#      ##      $#      %#      &#      '#      (#      )#      *#      +#      ,#      -#      .#      /#      0#      1#      2#      3#      4#      5#      6#      7#      8#      9#      :#      ;#      <#      =#      >#      ?#      @#      A#      B#      C#      D#      E#      F#      G#      H#      I#      J#      K#      L#      M#      N#      O#      P#      Q#      R#      S#      T#      U#      V#      W#      X#      Y#      Z#      [#      \#      ]#      ^#      _#      `#      a#      b#      c#      d#      e#      f#      g#      h#      i#      j#      k#      l#      m#      n#      o#      p#      q#      r#      s#      t#      u#      v#      w#      x#      y#      z#      {#      |#      }#      ~#      #      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#      ?#       $      $      $      $      $      $      $      $      $      	$      
$      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $      $       $      !$      "$      #$      $$      %$      &$      '$      ($      )$      *$      +$      ,$      -$      .$      /$      0$      1$      2$      3$      4$      5$      6$      7$      8$      9$      :$      ;$      <$      =$      >$      ?$      @$      A$      B$      C$      D$      E$      F$      G$      H$      I$      J$      K$      L$      M$      N$      O$      P$      Q$      R$      S$      T$      U$      V$      W$      X$      Y$      Z$      [$      \$      ]$      ^$      _$      `$      a$      b$      c$      d$      e$      f$      g$      h$      i$      j$      k$      l$      m$      n$      o$      p$      q$      r$      s$      t$      u$      v$      w$      x$      y$      z$      {$      |$      }$      ~$      $      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$      ?$       %      %      %      %      %      %      %      %      %      	%      
%      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %      %       %      !%      "%      #%      $%      %%      &%      '%      (%      )%      *%      +%      ,%      -%      .%      /%      0%      1%      2%      3%      4%      5%      6%      7%      8%      9%      :%      ;%      <%      =%      >%      ?%      @%      A%      B%      C%      D%      E%      F%      G%      H%      I%      J%      K%      L%      M%      N%      O%      P%      Q%      R%      S%      T%      U%      V%      W%      X%      Y%      Z%      [%      \%      ]%      ^%      _%      `%      a%      b%      c%      d%      e%      f%      g%      h%      i%      j%      k%      l%      m%      n%      o%      p%      q%      r%      s%      t%      u%      v%      w%      x%      y%      z%      {%      |%      }%      ~%      %      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%      ?%       &      &      &      &      &      &      &      &      &      	&      
&      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &      &       &      !&      "&      #&      $&      %&      &&      '&      (&      )&      *&      +&      ,&      -&      .&      /&      0&      1&      2&      3&      4&      5&      6&      7&      8&      9&      :&      ;&      <&      =&      >&      ?&      @&      A&      B&      C&      D&      E&      F&      G&      H&      I&      J&      K&      L&      M&      N&      O&      P&      Q&      R&      S&      T&      U&      V&      W&      X&      Y&      Z&      [&      \&      ]&      ^&      _&      `&      a&      b&      c&      d&      e&      f&      g&      h&      i&      j&      k&      l&      m&      n&      o&      p&      q&      r&      s&      t&      u&      v&      w&      x&      y&      z&      {&      |&      }&      ~&      &      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&      ?&       '      '      '      '      '      '      '      '      '      	'      
'      '      '      '      '      '      
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *"
fR
__inference_<lambda>_2094
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *"
fR
__inference_<lambda>_2099
8
NoOpNoOp^PartitionedCall^StatefulPartitionedCall
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api

signatures
"
_lookup_layer
	keras_api
 
 
 
?
	non_trainable_variables


layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 
3
lookup_table
token_counts
	keras_api
 
 

0
 
 
 

_initializer
LJ
tableAlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_input_1
hash_tableConstConst_1Const_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference_signature_wrapper_1841
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1Const_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference__traced_save_2136
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameMutableHashTable*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_restore_2149??
?
?
__inference_<lambda>_20947
3key_value_init1446_lookuptableimportv2_table_handle/
+key_value_init1446_lookuptableimportv2_keys1
-key_value_init1446_lookuptableimportv2_values	
identity??&key_value_init1446/LookupTableImportV2?
&key_value_init1446/LookupTableImportV2LookupTableImportV23key_value_init1446_lookuptableimportv2_table_handle+key_value_init1446_lookuptableimportv2_keys-key_value_init1446_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init1446/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2P
&key_value_init1446/LookupTableImportV2&key_value_init1446/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?W
?
D__inference_sequential_layer_call_and_return_conditional_losses_1919

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*(
_output_shapes
:???????????
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
-
__inference__initializer_2047
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?W
?
D__inference_sequential_layer_call_and_return_conditional_losses_1698

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*(
_output_shapes
:???????????
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_save_fn_2071
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?W
?
D__inference_sequential_layer_call_and_return_conditional_losses_1620

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*(
_output_shapes
:???????????
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
+
__inference__destroyer_2052
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_20327
3key_value_init1446_lookuptableimportv2_table_handle/
+key_value_init1446_lookuptableimportv2_keys1
-key_value_init1446_lookuptableimportv2_values	
identity??&key_value_init1446/LookupTableImportV2?
&key_value_init1446/LookupTableImportV2LookupTableImportV23key_value_init1446_lookuptableimportv2_table_handle+key_value_init1446_lookuptableimportv2_keys-key_value_init1446_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: o
NoOpNoOp'^key_value_init1446/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :?N:?N2P
&key_value_init1446/LookupTableImportV2&key_value_init1446/LookupTableImportV2:!

_output_shapes	
:?N:!

_output_shapes	
:?N
?
)
__inference_<lambda>_2099
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
"__inference_signature_wrapper_1841
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__wrapped_model_1564p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_1722
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1698p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?W
?
D__inference_sequential_layer_call_and_return_conditional_losses_1971

inputsO
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2^
text_vectorization/StringLowerStringLowerinputs*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*(
_output_shapes
:???????????
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?W
?
D__inference_sequential_layer_call_and_return_conditional_losses_1774
input_1O
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2_
text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*(
_output_shapes
:???????????
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_1867

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1698p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?C
?
__inference_adapt_step_2019
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*#
_output_shapes
:?????????*"
output_shapes
:?????????*
output_types
2]
StringLowerStringLowerIteratorGetNext:components:0*#
_output_shapes
:??????????
StaticRegexReplaceStaticRegexReplaceStringLower:output:0*#
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite R
StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
StringSplit/StringSplitV2StringSplitV2StaticRegexReplace:output:0StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:p
StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        r
!StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
StringSplit/strided_sliceStridedSlice#StringSplit/StringSplitV2:indices:0(StringSplit/strided_slice/stack:output:0*StringSplit/strided_slice/stack_1:output:0*StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_maskk
!StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
StringSplit/strided_slice_1StridedSlice!StringSplit/StringSplitV2:shape:0*StringSplit/strided_slice_1/stack:output:0,StringSplit/strided_slice_1/stack_1:output:0,StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
BStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast"StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast$StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdUStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
PStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterTStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0YStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
KStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastRStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
LStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2SStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0UStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulOStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumHStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
NStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
OStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountFStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0WStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumVStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
MStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
IStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
DStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2VStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0JStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0RStringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCounts"StringSplit/StringSplitV2:values:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_1854

inputs
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1620p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
 __inference__traced_restore_2149
file_prefixM
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: 

identity_1??2MutableHashTable_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?BFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2	?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:0RestoreV2:tensors:1*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
IdentityIdentityfile_prefix3^MutableHashTable_table_restore/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: S

Identity_1IdentityIdentity:output:0^NoOp_1*
T0*
_output_shapes
: }
NoOp_1NoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable
?_
?
__inference__wrapped_model_1564
input_1Z
Vsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle[
Wsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	7
3sequential_text_vectorization_string_lookup_equal_y:
6sequential_text_vectorization_string_lookup_selectv2_t	
identity	??Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2j
)sequential/text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:??????????
0sequential/text_vectorization/StaticRegexReplaceStaticRegexReplace2sequential/text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
%sequential/text_vectorization/SqueezeSqueeze9sequential/text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????p
/sequential/text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
7sequential/text_vectorization/StringSplit/StringSplitV2StringSplitV2.sequential/text_vectorization/Squeeze:output:08sequential/text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
=sequential/text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
?sequential/text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
?sequential/text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
7sequential/text_vectorization/StringSplit/strided_sliceStridedSliceAsequential/text_vectorization/StringSplit/StringSplitV2:indices:0Fsequential/text_vectorization/StringSplit/strided_slice/stack:output:0Hsequential/text_vectorization/StringSplit/strided_slice/stack_1:output:0Hsequential/text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
?sequential/text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/text_vectorization/StringSplit/strided_slice_1StridedSlice?sequential/text_vectorization/StringSplit/StringSplitV2:shape:0Hsequential/text_vectorization/StringSplit/strided_slice_1/stack:output:0Jsequential/text_vectorization/StringSplit/strided_slice_1/stack_1:output:0Jsequential/text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
`sequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast@sequential/text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastBsequential/text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapedsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0ssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
nsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreaterrsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0wsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
isequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastpsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxdsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0usequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
jsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2qsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0ssequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulmsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumfsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumfsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
lsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
msequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountdsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0usequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumtsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
ksequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
gsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
bsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2tsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0hsequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0psequential/text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Vsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle@sequential/text_vectorization/StringSplit/StringSplitV2:values:0Wsequential_text_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
1sequential/text_vectorization/string_lookup/EqualEqual@sequential/text_vectorization/StringSplit/StringSplitV2:values:03sequential_text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
4sequential/text_vectorization/string_lookup/SelectV2SelectV25sequential/text_vectorization/string_lookup/Equal:z:06sequential_text_vectorization_string_lookup_selectv2_tRsequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
4sequential/text_vectorization/string_lookup/IdentityIdentity=sequential/text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????|
:sequential/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
2sequential/text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
Asequential/text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor;sequential/text_vectorization/RaggedToTensor/Const:output:0=sequential/text_vectorization/string_lookup/Identity:output:0Csequential/text_vectorization/RaggedToTensor/default_value:output:0Bsequential/text_vectorization/StringSplit/strided_slice_1:output:0@sequential/text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentityJsequential/text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*(
_output_shapes
:???????????
NoOpNoOpJ^sequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2Isequential/text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference_restore_fn_2079
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?W
?
D__inference_sequential_layer_call_and_return_conditional_losses_1826
input_1O
Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(text_vectorization_string_lookup_equal_y/
+text_vectorization_string_lookup_selectv2_t	
identity	??>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2_
text_vectorization/StringLowerStringLowerinput_1*'
_output_shapes
:??????????
%text_vectorization/StaticRegexReplaceStaticRegexReplace'text_vectorization/StringLower:output:0*'
_output_shapes
:?????????*6
pattern+)[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']*
rewrite ?
text_vectorization/SqueezeSqueeze.text_vectorization/StaticRegexReplace:output:0*
T0*#
_output_shapes
:?????????*
squeeze_dims

?????????e
$text_vectorization/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B ?
,text_vectorization/StringSplit/StringSplitV2StringSplitV2#text_vectorization/Squeeze:output:0-text_vectorization/StringSplit/Const:output:0*<
_output_shapes*
(:?????????:?????????:?
2text_vectorization/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
4text_vectorization/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
4text_vectorization/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
,text_vectorization/StringSplit/strided_sliceStridedSlice6text_vectorization/StringSplit/StringSplitV2:indices:0;text_vectorization/StringSplit/strided_slice/stack:output:0=text_vectorization/StringSplit/strided_slice/stack_1:output:0=text_vectorization/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask~
4text_vectorization/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6text_vectorization/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6text_vectorization/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.text_vectorization/StringSplit/strided_slice_1StridedSlice4text_vectorization/StringSplit/StringSplitV2:shape:0=text_vectorization/StringSplit/strided_slice_1/stack:output:0?text_vectorization/StringSplit/strided_slice_1/stack_1:output:0?text_vectorization/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask?
Utext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5text_vectorization/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:??????????
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7text_vectorization/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: ?
ctext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0ltext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
^text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastetext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: ?
_text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2ftext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0htext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: ?
]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: ?
atext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 ?
btext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:??????????
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumitext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:??????????
`text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R ?
\text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Wtext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2itext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]text_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0etext_vectorization/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:??????????
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Ktext_vectorization_string_lookup_none_lookup_lookuptablefindv2_table_handle5text_vectorization/StringSplit/StringSplitV2:values:0Ltext_vectorization_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:??????????
&text_vectorization/string_lookup/EqualEqual5text_vectorization/StringSplit/StringSplitV2:values:0(text_vectorization_string_lookup_equal_y*
T0*#
_output_shapes
:??????????
)text_vectorization/string_lookup/SelectV2SelectV2*text_vectorization/string_lookup/Equal:z:0+text_vectorization_string_lookup_selectv2_tGtext_vectorization/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
)text_vectorization/string_lookup/IdentityIdentity2text_vectorization/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:?????????q
/text_vectorization/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
'text_vectorization/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"?????????       ?
6text_vectorization/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0text_vectorization/RaggedToTensor/Const:output:02text_vectorization/string_lookup/Identity:output:08text_vectorization/RaggedToTensor/default_value:output:07text_vectorization/StringSplit/strided_slice_1:output:05text_vectorization/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*(
_output_shapes
:??????????*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS?
IdentityIdentity?text_vectorization/RaggedToTensor/RaggedTensorToTensor:result:0^NoOp*
T0	*(
_output_shapes
:???????????
NoOpNoOp?^text_vectorization/string_lookup/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2?
>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2>text_vectorization/string_lookup/None_Lookup/LookupTableFindV2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
9
__inference__creator_2024
identity??
hash_tablel

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name1447*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
+
__inference__destroyer_2037
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
E
__inference__creator_2042
identity: ??MutableHashTable}
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_97*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?
?
)__inference_sequential_layer_call_fn_1631
input_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2		*
Tout
2	*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1620p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
__inference__traced_save_2136
file_prefixJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?BFlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-keysBHlayer_with_weights-0/_lookup_layer/token_counts/.ATTRIBUTES/table-valuesB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1savev2_const_6"/device:CPU:0*
_output_shapes
 *
dtypes
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
::

_output_shapes
::

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????I
text_vectorization3
StatefulPartitionedCall_1:0	??????????tensorflow/serving/predict:?6
?
layer_with_weights-0
layer-0
	variables
trainable_variables
regularization_losses
	keras_api

signatures
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_sequential
P
_lookup_layer
	keras_api
_adapt_function"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	non_trainable_variables


layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
L
lookup_table
token_counts
	keras_api"
_tf_keras_layer
"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
j
_initializer
_create_resource
_initialize
_destroy_resourceR jCustom.StaticHashTable
O
_create_resource
_initialize
_destroy_resourceR Z
table
"
_generic_user_object
"
_generic_user_object
?2?
)__inference_sequential_layer_call_fn_1631
)__inference_sequential_layer_call_fn_1854
)__inference_sequential_layer_call_fn_1867
)__inference_sequential_layer_call_fn_1722?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_1919
D__inference_sequential_layer_call_and_return_conditional_losses_1971
D__inference_sequential_layer_call_and_return_conditional_losses_1774
D__inference_sequential_layer_call_and_return_conditional_losses_1826?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
__inference__wrapped_model_1564input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_adapt_step_2019?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_1841input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference__creator_2024?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_2032?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2037?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_2042?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_2047?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_2052?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
__inference_save_fn_2071checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_2079restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_55
__inference__creator_2024?

? 
? "? 5
__inference__creator_2042?

? 
? "? 7
__inference__destroyer_2037?

? 
? "? 7
__inference__destroyer_2052?

? 
? "? >
__inference__initializer_2032#$?

? 
? "? 9
__inference__initializer_2047?

? 
? "? ?
__inference__wrapped_model_1564? !0?-
&?#
!?
input_1?????????
? "H?E
C
text_vectorization-?*
text_vectorization??????????	h
__inference_adapt_step_2019I"??<
5?2
0?-?
??????????IteratorSpec 
? "
 x
__inference_restore_fn_2079YK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_2071?&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
D__inference_sequential_layer_call_and_return_conditional_losses_1774h !8?5
.?+
!?
input_1?????????
p 

 
? "&?#
?
0??????????	
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1826h !8?5
.?+
!?
input_1?????????
p

 
? "&?#
?
0??????????	
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1919g !7?4
-?*
 ?
inputs?????????
p 

 
? "&?#
?
0??????????	
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1971g !7?4
-?*
 ?
inputs?????????
p

 
? "&?#
?
0??????????	
? ?
)__inference_sequential_layer_call_fn_1631[ !8?5
.?+
!?
input_1?????????
p 

 
? "???????????	?
)__inference_sequential_layer_call_fn_1722[ !8?5
.?+
!?
input_1?????????
p

 
? "???????????	?
)__inference_sequential_layer_call_fn_1854Z !7?4
-?*
 ?
inputs?????????
p 

 
? "???????????	?
)__inference_sequential_layer_call_fn_1867Z !7?4
-?*
 ?
inputs?????????
p

 
? "???????????	?
"__inference_signature_wrapper_1841? !;?8
? 
1?.
,
input_1!?
input_1?????????"H?E
C
text_vectorization-?*
text_vectorization??????????	