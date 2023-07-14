# Description

__Training__

Our earlier JLF-based work utilized a set of 17 hand-labeled atlases which were
annotated using T2-weighted imaging (with accompanying T1 image data) based on a
validated, well-established protocol [@Reagh:2018aa]. T1 MPRAGE [@Mugler:1990vk]
scans were acquired in the sagittal orientation with an isotropic image
resolution of $0.75 \times 0.75 \times 0.75$ mm$^3$. T2 image acquisition was
performed perpendicular to the long axis of the hippocampus with resolution
$0.47 \times 0.47 \times 2.0$ mm$^3$ consistent with previous recommendations
[@Yushkevich:2010aa].  The optimal rigid transformation between corresponding T1
and T2 images was determined using the Advanced Normalization Tools (ANTs)
software package [@Avants:2014aa].

\begin{figure}
  \centering
    \includegraphics{Figures/deepFlashTemplate.png}
    \caption{Coronal (left) and sagittal (right) views of the MTL parcellation generated
    from DeepFLASH superimposed on the T1 template component (T2 not shown).
    The template pose is oriented analogously to hippocampal specific MR acquisition
    protocols for a tailored segmentation domain.}
    \label{fig:deepFlashTemplate}
\end{figure}

These 17 image pairs and their contralateral counterparts were used to create an
optimal symmetric multivariate template [@Avants:2010aa;@Tustison:2015vl] in the
orientation of the original T2 acquisition (i.e., perpendicular to the
hippocampal axis) as illustrated in Figure \ref{fig:deepFlashTemplate}.  Each of
the pairwise normalized images and corresponding labels were subsequently
affinely registered to the symmetric template where the labels were used to
create average prior probability images to be used as additional channel inputs
for training/prediction. The ROIs for training/prediction for both left and right
hippocampi are contralaterally symmetric within the template domain and are of
size $64 \times 64 \times 96$ voxels (voxel size = $1 mm^3$).

_Intensity-based data augmentation_ consisted of randomly added noise based on
ITK functionality, simulated bias fields based on N4 bias field modeling
[@Tustison:2010ac], and histogram warping for mimicking well-known MRI
intensity nonlinearities [@Nyul:1999th;@Tustison:2021ab]. These augmentation
techniques are available in ANTsXNet (Python and R):

* simulated bias field
    * ``simulate_bias_field.py``
    * ``simulateBiasField.R``
* added noise
    * ``add_noise_to_image.py``
    * ``addNoiseToImage.R``
* MRI intensity nonlinearities
    * ``histogram_warp_image_intensities.py``
    * ``histogramWarpImageIntensities.R``

_Shape-based data augmentation_ used both random linear and nonlinear
deformations after whole-brain affine normalization to the template.  This
functionality is also instantiated within ANTsXNet:

* random spatial warping
    * ``randomly_transform_image_data.py``
    * ``randomlyTransformImageData.R``

Additionally, given the well-studied variation of hippocampal volume with age
[@Driscoll:2003vj;@Bussy:2021uk] and disease (e.g., [@Henneman:2009vk]), the
original 17 image sets were warped to additional age/gender templates created
from the public IXI database (i.e., 20-30, 30-40, 50-60, 70-80, 80-90 and
male/female) and ADNI templates made from diagnostic cohorts (i.e., normal, mild
cognitive impairment, and Alzheimer's disease).  This resulted in an extended
training data set of an additional 288 subjects ($\times 2$ by including the
contralateral versions).  Prior to training, these data were skull stripped
using ANTsXNet functionality and affinely warped to a brain
extraction version of the DeepFLASH templates. From these training data, random
intensity and shape data augmentations, as previously described, were generated
on the fly over 200 epochs with a batch size of 16.

During the early training stages, a combined loss function incorporated the
weighted categorical entropy for the multilabel first output and the binary dice
coefficient for the remaining binary label regional outputs.  Later stages
replaced the weighted categorical entropy with a multilabel dice coefficient
loss for the first output (with other output loss functions remaining the same).
Additionally, earlier stages weighted more heavily the contribution of the
binary regional outputs over the multilabel first output which was reversed in
later stages.  All training was performed on a DGX (GPUs: 4X Tesla V100, system
memory: 256 GB LRDIMM DDR4) using ANTsXNet functionality with the training
scripts available with the GitHub repository associated with this work [@antsxukbb].

__Voxelwise Label Prediction__

Given the bilateral symmetry in both the training data and training protocol,
two sets of networks, left and right, were trained using both T1 and combined
T1/T2 modalities.  This symmetry prevents a prior volumetric bias from being
encoded within the segmentation framework which limits biasing of hemispherical
asymmetry in hippocampal volumes under conditions of aging and pathology
[@Thompson:2009tr;@Rogers:2012vt;@Sarica:2018wj].  An additional advantage is
that two trained networks are available for predicting both MTL regions by
contralaterally flipping the results and averaging the estimated probability
maps.

For predicting the MTL parcellation, preprocessing includes N4 bias correction
[@Tustison:2010ac], brain extraction [@Tustison:2021ab], affine warping
[@Avants:2011wx;@Avants:2014aa] to the brain extracted DeepFLASH template, and
cropping to the hippocampal ROIs in the space of the template.  For intensity
matching within the respective ROIs, two different protocols were explored:
rank intensity transform of the input images (e.g., intensity values of [0, 1,
2, 255] are converted to [0, 1, 2, 3]) and histogram matching [@Nyul:1999th] to
the MTL region of the corresponding modality template.  Although producing
similar results in our preliminary qualitative evaluations using a wide dataset
sampling, we use the former for UKBB.  However, both options are available to
the user. During prediction, the U-net network is constructed internally and the
requisite additional data (i.e., parameter weights and templates) are
automatically downloaded from Figshare and cached using TensorFlow/Keras
utilities.

