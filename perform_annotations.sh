#!/bin/bash

cd /media/ruppenhofer/erbat/ASLAN/forks/preprocessing

XMIDIR=$1
LANG=$2


ALL_VIEWS="_InitialView wtpsplit_segmented rwse_normalized spelling_normalized nominal_ellipsis_resolved vpe_resolved coord_subjects_explicated gapped_coordination_resolved coref_normalized"




# echo "add_stanza_parses.py"
python3 add_stanza_parses.py  $XMIDIR --language ${LANG} --view ${ALL_VIEWS}

echo "\n"

echo "add_spelling_errors: needs tokenization, so after parsing"
python3 add_spelling_errors.py  --language ${LANG} --view _InitialView wtpsplit_segmented rwse_normalized -- $XMIDIR



echo "\n"
echo "add_subject_sharing.py: assumes that the view has parses"
python3 add_subject_sharing.py  --view _InitialView wtpsplit_segmented rwse_normalized spelling_normalized nominal_ellipsis_resolved vpe_resolved --lang ${LANG} -- $XMIDIR

echo "\n"
python3 add_passive.py --language ${LANG} --view ${ALL_VIEWS} -- $XMIDIR 


echo "\n"
echo "add_clefts.py: assumes view has parses"
python3 add_clefts.py  --lang ${LANG} --view ${ALL_VIEWS} -- $XMIDIR

echo "\n"
echo "add_nominal_ellipsis.py: assumes view has parses"
python add_nominal_ellipsis.py --view _InitialView wtpsplit_segmented rwse_normalized spelling_normalized --lang ${LANG} -- $XMIDIR

echo "\n"
echo "add_verbal_ellipsis.py: assumes view has parses"
python add_verbal_ellipsis.py --lang ${LANG} --view _InitialView wtpsplit_segmented rwse_normalized spelling_normalized nominal_ellipsis_resolved -- $XMIDIR

echo "\n"
python add_gapped_coordination.py --lang ${LANG} --view _InitialView wtpsplit_segmented rwse_normalized spelling_normalized nominal_ellipsis_resolved vpe_resolved coord_subjects_explicated --  $XMIDIR


#echo "\n"
#echo "add_sluicing.py: assumes view has parses"
#python3 add_sluicing.py  $XMIDIR --view ${ALL_VIEWS}


echo "\n"
echo "add_coreference.py: in present form , now assumes the presence of a parse (I think)"
poetry run python3 add_coreference.py --language ${LANG} --view _InitialView wtpsplit_segmented rwse_normalized spelling_normalized nominal_ellipsis_resolved vpe_resolved coord_subjects_explicated gapped_coordination_resolved -- $XMIDIR 


