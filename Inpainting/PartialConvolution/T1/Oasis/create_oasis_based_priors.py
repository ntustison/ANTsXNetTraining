import ants
import glob
import numpy as np

base_directory = "/Users/ntustison/Data/Public/"

template = ants.image_read("~/Desktop/Oasis/deepAtropos.nii.gz")

atropos_files = (*glob.glob(base_directory + "CorticalThicknessData2014/*/DeepAtropos/*.nii.gz"),
                 *glob.glob(base_directory + "SRPB1600/processed_data/sub-*/t1/defaced_mprageDeepAtropos.nii.gz"))

priors = list()
for i in range(6):
    prior_array = np.zeros(template.shape)
    prior_image = ants.from_numpy(prior_array, origin=template.origin,
        spacing=template.spacing, direction=template.direction)
    priors.append(prior_image)

for i in range(len(atropos_files)):
    print(str(i) + " out of " + str(len(atropos_files)))

    seg = ants.image_read(atropos_files[i])

    center_of_mass_template = ants.get_center_of_mass(template)
    center_of_mass_image = ants.get_center_of_mass(seg)
    translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
    xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
        center=np.asarray(center_of_mass_template), translation=translation)
    seg = xfrm.apply_to_image(seg, template)

    for j in range(len(priors)):
        tissue = ants.threshold_image(seg, j+1, j+1, 1, 0)
        priors[j] = (priors[j] * i + tissue) / (i + 1)
        if i % 10 == 0:
            ants.image_write(priors[j], "/Users/ntustison/Desktop/Oasis/priors" + str(j+1) + ".nii.gz")

