from rembg import remove 
from PIL import Image
import cv2
import numpy as np
import os

counter = 0

def main():
   class_1 = "data/classification/Achnatherum_inebrians"
   class_2 = "data/classification/Achnatherum_splendens"
   class_3 = "data/classification/Agriophyllum_squarrosum"
   class_4 = "data/classification/Agropyron_cristatum"
   class_5 = "data/classification/Agropyron_elongatum"
   class_6 = "data/classification/Agropyron_mongolicum"
   class_7 = "data/classification/Amorpha_fruticosa"
   class_8 = "data/classification/Amygdalus_mongolica"
   class_9 = "data/classification/Anemone_rivularis"
   class_10 = "data/classification/Apocynum_pictum"
   class_11 = "data/classification/Apocynum_venetum"
   class_12 = "data/classification/Artemisia_desertorum"
   class_13 = "data/classification/Artemisia_ordosica"
   class_14 = "data/classification/Artemisia_sphaerocephala"
   class_15 = "data/classification/Astragalus_laxmannii"
   class_16 = "data/classification/Astragalus_melilotoides"
   class_17 = "data/classification/Atriplex_sibirica"
   class_18 = "data/classification/Avena_sativa"
   class_19 = "data/classification/Bassia_dasyphylla"
   class_20 = "data/classification/Bromus_inermis"
   class_21 = "data/classification/Buddleja_alternifolia"
   class_22 = "data/classification/Calligonum_mongolicum"
   class_23 = "data/classification/Calligonum_mongolicum_Turcz"
   class_24 = "data/classification/Caragana_korshinskii"
   class_25 = "data/classification/Caragana_liouana"
   class_26 = "data/classification/Caragana_microphylla"
   class_27 = "data/classification/Caryopteris_mongholica"
   class_28 = "data/classification/Clematis_fruticosa"
   class_29 = "data/classification/Corethrodendron_fruticosum_mongolicum"
   class_30 = "data/classification/Corethrodendron_lignosum"
   class_31 = "data/classification/Corethrodendron_scoparium"
   class_32 = "data/classification/Coronilla_varia"
   class_33 = "data/classification/Cosmos_bipinnatus"
   class_34 = "data/classification/Cynanchum_acutum"
   class_35 = "data/classification/Descurainia_sophia"
   class_36 = "data/classification/Elaeagnus_angustifolia"
   class_37 = "data/classification/Elymus_nutans"
   class_38 = "data/classification/Elymus_pendulinus_pubicaulis"
   class_39 = "data/classification/Elymus_sibiricus"
   class_40 = "data/classification/Euphrasia_pectinata"
   class_41 = "data/classification/Festuca_rubra"
   class_42 = "data/classification/Glycyrrhiza_glabra"
   class_43 = "data/classification/Halimodendron_halodendron"
   class_44 = "data/classification/Halogeton_arachnoideus"
   class_45 = "data/classification/Halostachys_caspica"
   class_46 = "data/classification/Haloxylon_ammodendron"
   class_47 = "data/classification/Hedysarum_scoparium"
   class_48 = "data/classification/Hippophae_rhamnoides"
   class_49 = "data/classification/Hordeum_vulgare"
   class_50 = "data/classification/Hyoscyamus_niger"
   class_51 = "data/classification/Iris_lactea"
   class_52 = "data/classification/Kochia_scoparia"
   class_53 = "data/classification/Leonurus_japonicus"
   class_54 = "data/classification/Lespedeza_bicolor"
   class_55 = "data/classification/Lespedeza_daurica"
   class_56 = "data/classification/Leymus_chinensis"
   class_57 = "data/classification/Ligularia_virgaurea"
   class_58 = "data/classification/Limonium_aureum"
   class_59 = "data/classification/Lolium_perenne"
   class_60 = "data/classification/Lycium_chinese"
   class_61 = "data/classification/Lycium_ruthenicum"
   class_62 = "data/classification/Medicago_lupulina"
   class_63 = "data/classification/Medicago_sativa"
   class_64 = "data/classification/Melilotus_officinalis"
   class_65 = "data/classification/Nitraria_sibirica"
   class_66 = "data/classification/Nitraria_tangutorum"
   class_67 = "data/classification/Oxytropis_bicolor"
   class_68 = "data/classification/Peganum_harmala"
   class_69 = "data/classification/Picea_asperata"
   class_70 = "data/classification/Pisum_sativum_L"
   class_71 = "data/classification/Plantago_major"
   class_72 = "data/classification/Poa_annua"
   class_73 = "data/classification/Puccinellia_distans"
   class_74 = "data/classification/Puccinellia_tenuiflora"
   class_75 = "data/classification/Reaumuria_songarica"
   class_76 = "data/classification/Rumex_nepalensis"
   class_77 = "data/classification/Rumex_patientia"
   class_78 = "data/classification/Salsola_tragus"
   class_79 = "data/classification/Saposhnikovia_divaricata"
   class_80 = "data/classification/Setaria_viridis"
   class_81 = "data/classification/Stipa_bungeana"
   class_82 = "data/classification/Thermopsis_lanceolata"
   class_83 = "data/classification/Thlaspi_arvense"
   class_84 = "data/classification/Trifolium_repens"
   class_85 = "data/classification/Triticale"
   class_86 = "data/classification/Vicia_sativa"
   class_87 = "data/classification/Vicia_villosa"
   class_88 = "data/classification/Zygophyllum_xanthoxylon"

   creator_v2(class_1)
   creator_v2(class_2)
   creator_v1(class_3)
   creator_v1(class_4)
   creator_v1(class_5)
   creator_v1(class_6)
   creator_v1(class_7)
   creator_v1(class_8)
   creator_v1(class_9)
   creator_v1(class_10)
   creator_v2(class_11)
   creator_v2(class_12)
   creator_v2(class_13)
   creator_v1(class_14)
   creator_v1(class_15)
   creator_v1(class_16)
   creator_v1(class_17)
   creator_v1(class_18)
   creator_v2(class_19)
   creator_v1(class_20)
   creator_v2(class_21)
   creator_v2(class_22)
   creator_v2(class_23)
   creator_v1(class_24)
   creator_v1(class_25)
   creator_v1(class_26)
   creator_v1(class_27)
   creator_v2(class_28)
   creator_v1(class_29)
   creator_v1(class_30)
   creator_v1(class_31)
   creator_v1(class_32)
   creator_v2(class_33)
   creator_v2(class_34)
   creator_v2(class_35)
   creator_v1(class_36)
   creator_v2(class_37)
   creator_v2(class_38)
   creator_v1(class_39)
   creator_v2(class_40)
   creator_v1(class_41)
   creator_v1(class_42)
   creator_v1(class_43)
   creator_v2(class_44)
   creator_v1(class_45)
   creator_v1(class_46)
   creator_v1(class_47)
   creator_v1(class_48)
   creator_v1(class_49)
   creator_v2(class_50)
   creator_v1(class_51)
   creator_v2(class_52)
   creator_v1(class_53)
   creator_v2(class_54)
   creator_v1(class_55)
   creator_v1(class_56)
   creator_v2(class_57)
   creator_v2(class_58)
   creator_v1(class_59)
   creator_v2(class_60)
   creator_v1(class_61)
   creator_v1(class_62)
   creator_v1(class_63)
   creator_v1(class_64)
   creator_v1(class_65)
   creator_v1(class_66)
   creator_v1(class_67)
   creator_v1(class_68)
   creator_v1(class_69)
   creator_v2(class_70)
   creator_v2(class_71)
   creator_v1(class_72)
   creator_v2(class_73)
   creator_v2(class_74)
   creator_v1(class_75)
   creator_v2(class_76)
   creator_v1(class_77)
   creator_v2(class_78)
   creator_v2(class_79)
   creator_v2(class_80)
   creator_v2(class_81)
   creator_v1(class_82)
   creator_v1(class_83)
   creator_v2(class_84)
   creator_v1(class_85)
   creator_v1(class_86)
   creator_v1(class_87)
   creator_v1(class_88)



def creator_v1(path):
   images = os.listdir(path)
   for i in images:
      img_path = path + "/" + i
      global counter
      out_path = "data/seedOnly/img" + str(counter) + ".png"
      bg_remover(img_path, out_path)
      print("Processed Image Number: " + str(counter))
      counter = counter + 1

def creator_v2(path):
   images = os.listdir(path)
   for i in images:
      img_path = path + "/" + i
      global counter
      out_path = "data/seedOnly/img" + str(counter) + ".png"
      bg_remover_v2(img_path, out_path)
      print("Processed Image Number: " + str(counter))
      counter = counter + 1

def bg_remover(in_path, out_path):

  # Processing the image 
  input = Image.open(in_path) 
    
  # Removing the background from the given Image 
  output = remove(input) 
    
  #Saving the image in the given path 
  output.save(out_path)   


def bg_remover_v2(in_path, out_path):
  # Load the seed image
  img = cv2.imread(in_path)  # Replace with your image file path
  if img is None:
      raise FileNotFoundError("The image file was not found.")

  # Create a mask initialized to zero (definite background)
  mask = np.zeros(img.shape[:2], np.uint8)

  # Allocate arrays for background and foreground models (used by GrabCut internally)
  bgdModel = np.zeros((1, 65), np.float64)
  fgdModel = np.zeros((1, 65), np.float64)

  # Define a rectangle that includes the seed.
  # You might need to adjust these values to tightly enclose the seed.
  height, width = img.shape[:2]
  rect = (10, 10, width - 20, height - 20)

  # Run GrabCut algorithm. It modifies the mask such that:
  # - 0 and 2 indicate background
  # - 1 and 3 indicate foreground (seed)
  cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

  # Convert mask: pixels marked as 0 or 2 will be set to 0 (background); 1 or 3 to 1 (foreground)
  mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

  # Multiply the original image with the mask to extract the foreground
  img_foreground = img * mask2[:, :, np.newaxis]

  # Convert image to BGRA (adds an alpha channel)
  img_rgba = cv2.cvtColor(img_foreground, cv2.COLOR_BGR2BGRA)

  # Set background pixels (where mask is 0) to fully transparent
  img_rgba[mask2 == 0] = (0, 0, 0, 0)

  # Save the output image with transparent background
  cv2.imwrite(out_path, img_rgba)

if __name__ == "__main__":
    main()