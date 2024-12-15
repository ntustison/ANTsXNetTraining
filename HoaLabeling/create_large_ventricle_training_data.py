import ants
import glob
import os

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = "4"

base_directory = "/Users/ntustison/Data/Public/HCP/DeepLearning/HOATraining/"
data_directory = base_directory + "T1w/"
output_directory = base_directory + "T1w-LargeVentricles/"

t1_files = glob.glob(data_directory + "*T1w.nii.gz")

for i in range(len(t1_files)):
    print("Processing ", t1_files[i]) 
    print(str(i) + " out of " + str(len(t1_files)))

    t1_file = t1_files[i]
    da_file = t1_file.replace("T1w.nii.gz", "T1w_deep_atropos.nii.gz")
    bext_file = t1_file.replace("T1w.nii.gz", "T1w_bw20.nii.gz")
    hoa_file = t1_file.replace("T1w.nii.gz", "dseg.nii.gz").replace("T1w", "SubcorticalParcellations/dseg/")
    hoa_file2 = t1_file.replace("T1w.nii.gz", "dseg2.nii.gz").replace("T1w", "SubcorticalParcellations/dseg2/")

    output_prefix = output_directory + os.path.basename(t1_file).replace("T1w.nii.gz", "")

    bext = ants.image_read(bext_file)
    bext[bext == 1] = 0

    da = ants.image_read(da_file)
    da[da == 1] = 0
    da[da == 3] = 0
    da[da == 4] = 0

    hoa = ants.image_read(hoa_file)
    hoa_ventricles = ants.threshold_image(hoa, 1, 2, 1, 0)
    hoa_ventricles_dilation = ants.iMath_MD(hoa_ventricles, radius=8)

    reg = ants.label_image_registration([bext, da, hoa_ventricles_dilation],
                                        [bext, da, hoa_ventricles],
                                        type_of_linear_transform='identity',
                                        type_of_deformable_transform='antsRegistrationSyNQuick[so]',
                                        label_image_weighting=[1.0, 1.0, 10.0],
                                        output_prefix=output_prefix,
                                        verbose=True)
    
    t1 = ants.image_read(t1_file)
    t1_warped = ants.apply_transforms(t1, t1, reg['fwdtransforms'], interpolator="linear")
    ants.image_write(t1_warped, output_prefix + "lv_T1w.nii.gz")
    hoa_warped = ants.apply_transforms(t1, hoa, reg['fwdtransforms'], interpolator="genericLabel")
    ants.image_write(hoa_warped, output_prefix + "lv_dseg.nii.gz")

     
    hoa2 = ants.image_read(hoa_file2)
    mask = ants.threshold_image(hoa2, 0, 0, 0, 1)

    t1_warped_file = output_prefix + "lv_T1w.nii.gz"
    t1_warped = ants.image_read(t1_warped_file)

    t1_warped_extracted = t1_warped * mask
    ants.image_write(t1_warped_extracted, output_prefix + "lv_T1w_extracted.nii.gz")
    
    hoa_warped_file = output_prefix + "lv_dseg.nii.gz"
    hoa_warped = ants.image_read(hoa_warped_file)

    hoa_warped_plus = ants.threshold_image(hoa_warped, 0, 0, 1, 0) * hoa2 + hoa_warped
    ants.image_write(hoa_warped_plus, output_prefix + "lv_dseg2.nii.gz")
          
    
    
    
     
