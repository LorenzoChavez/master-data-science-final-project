# Dataset Download and Reconciliation
## -Not to be evaluated in TFM-

The purpose of these notebooks is to download the street2shop dataset and explore the challenges arising from this action.

One of the main challenges we are facing is due to some websites having antiscraping and antiscripting protections.
This introduces a random component when it comes to downloading the same set of images and forces us to follow different strategies that are hardly reproducible.

Other related challenges faced here include broken URLs, rectricted access to certain websites for EU individuals, placeholder images for expired items and corrupted images.

The goal in the notebooks within this directory is to collect a list of URLs to all the missing images in the final set and get in touch with the dataset owners. This route has been explored however, without success.

On the other hand, a final notebook solves the challenge due to the scale of the dataset by sizing it down from ~200GB to 19GB. This help us keep the dataset manageable while still preserving high quality data.

All things together, we consider this part troublesome when it comes to reproducibility therefore a reduced dataset will be provided for final evaluation on this project as well as another starting point to evaluate reproducibility.
