#! /usr/bin/sh

baseDirectory=/Users/ntustison/Data/HeliumLungStudies2/ProtonMasks/DeepLearning/
dataDirectory=${baseDirectory}/DataInTemplateSpace/
template=${baseDirectory}/protonLungTemplate.nii.gz

sumPrior=${baseDirectory}/sumPriors.nii.gz  
CreateImage 3 $template $sumPrior 0

for i in 1 2 3 4 5;
  do
    for j in `ls ${dataDirectory}/*Lobes.nii.gz`; 
      do
        basePrefix=`basename $j`
        localDir=`dirname $j`
        tmpImage=${localDir}/tmp_${basePrefix}_${i}.nii.gz
        `ThresholdImage 3 $j $tmpImage $i $i 1 0`
      done
    prior=${baseDirectory}/prior${i}.nii.gz  
    AverageImages 3 $prior 0 ${dataDirectory}/tmp_*_${i}.nii.gz
    ImageMath 3 $prior RescaleImage $prior 0 1
    SmoothImage 3 $prior 1.0 $prior 1
    ImageMath 3 $sumPrior + $prior $sumPrior

    rm -f ${dataDirectory}/tmp_*_${i}.nii.gz 
  done

ImageMath 3 $sumPrior RescaleImage $sumPrior 0 1

prior0=${baseDirectory}/prior0.nii.gz  
ImageMath 3 $prior0 m $sumPrior -1
ImageMath 3 $prior0 + $prior0 1

ImageMath 3 $sumPrior + $prior0 $sumPrior

for i in 0 1 2 3 4 5;
  do
    prior=${baseDirectory}/prior${i}.nii.gz  
    ImageMath 3 $prior / $prior $sumPrior
  done

