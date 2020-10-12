# XXX: competition name
COMPETITION := dacon-dku

# gsed on macOS. sed on LINUX
SED := gsed

# directories
DIR_DATA := data/$(COMPETITION)
DIR_BUILD := build
DIR_FEATURE := $(DIR_BUILD)/feature
DIR_METRIC := $(DIR_BUILD)/metric
DIR_MODEL := $(DIR_BUILD)/model

# directories for the cross validation and ensembling
DIR_VAL := $(DIR_BUILD)/val
DIR_TST := $(DIR_BUILD)/tst
DIR_SUB := $(DIR_BUILD)/sub

DIRS := $(DIR_DATA) $(DIR_BUILD) $(DIR_FEATURE) $(DIR_METRIC) $(DIR_MODEL) \
        $(DIR_VAL) $(DIR_TST) $(DIR_SUB)

# data files for training and predict
DATA_TRN := $(DIR_DATA)/train.csv
DATA_TST := $(DIR_DATA)/test.csv
SAMPLE_SUBMISSION := $(DIR_DATA)/sample_submission.csv

LABEL_IDX = 20

ID_TST := $(DIR_DATA)/id.tst.csv
HEADER := $(DIR_DATA)/header.csv

Y_TRN:= $(DIR_FEATURE)/y.trn.txt
Y_TST:= $(DIR_FEATURE)/y.tst.txt

data: $(DATA_TRN) $(DATA_TST) $(SAMPLE_SUBMISSION)

$(DIRS):
	mkdir -p $@

$(HEADER): $(SAMPLE_SUBMISSION)
	head -1 $< > $@

$(ID_TST): $(SAMPLE_SUBMISSION)
	cut -d, -f1 $< | tail -n +2 > $@

$(Y_TST): $(SAMPLE_SUBMISSION) | $(DIR_FEATURE)
	cut -d, -f2 $< | tail -n +2 > $@

$(Y_TRN): $(DATA_TRN) | $(DIR_FEATURE)
	cut -d, -f$(LABEL_IDX) $< | tail -n +2 > $@

# cleanup
clean::
	find . -name '*.pyc' -delete

clobber: clean
	-rm -rf $(DIR_DATA) $(DIR_BUILD)

.PHONY: clean clobber mac.setup ubuntu.setup apt.setup pip.setup
