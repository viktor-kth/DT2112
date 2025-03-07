from pathlib import Path
import pandas as pd
import json
import re
import os

import wptools


def get_wiki_info(name: str) -> dict:
    """
    Search Wikipedia for a person's name and return their image URL and description.
    
    Args:
        name (str): Name of the person to search for
    
    Returns:
        dict: Dictionary containing image_url and description, or None values if not found
    """
    try:
        # Clean the name
        clean_name = name
        
        # Get page info
        page = wptools.page(clean_name)
        page.get()
        
        result = {
            'image_url': None,
            'wiki_description': None,
            'url': None,
        }
        
        # Get the first image URL if available
        if hasattr(page, 'data') and 'image' in page.data:
            images = page.data['image']
            if images and len(images) > 0:
                # Get the first image that's not an icon or logo
                for img in images:
                    if isinstance(img, dict) and 'url' in img:
                        img = img['url']
                    if isinstance(img, str) and any(ext in img.lower() for ext in ['.jpg', '.jpeg', '.png']):
                        result['image_url'] = img
                        break
        else:
            result['image_url'] = ""
        
        # Get the description 
        if hasattr(page, 'data') and 'extract' in page.data:
            description = page.data['extract'][:500]
            description = description.replace('<p>', '').replace('</p>', '')
            description = description.replace('<b>', '').replace('</b>', '')
            description = description.replace('<i>', '').replace('</i>', '')
            description = description[28:] + "..."
            result['wiki_description'] = ' '.join(description.split())
        else: 
            result['wiki_description'] = ""
        
         # Get the Wikipedia URL
        if hasattr(page, 'data') and 'url' in page.data:
            result['wiki_url'] = page.data['url']
        else:
            result['wiki_url'] = ""
        

        print(f"Processed {clean_name}: {'Found' if result['wiki_description'] else 'No'} description, {'Found' if result['image_url'] else 'No'} image")
        
        # Add a small delay to avoid hitting rate limits
        #time.sleep(0.5)
        return result
        
    except Exception as e:
        print(f"Error searching Wikipedia for {name}: {e}")
        return {'image_url': "", 'wiki_description': "", 'wiki_url':""}

def append_data():
    cache_path = "./data/celeb_data_with_wiki.csv"
    og = ""
    if os.path.exists(cache_path):
        og = pd.read_csv(cache_path)
    
    df_videos = pd.read_csv("./data/celebs_dev.csv")

    og = og.drop(columns=['video','start_time'])
    df_videos = df_videos.drop(columns=['end_time','length', 'txt_file'])
    
    merged_df = og.merge(
        df_videos[['speaker', 'video', 'start_time']],
        left_on='VoxCeleb2_ID',
        right_on='speaker',
        how='left'
    ).drop_duplicates(subset=['VoxCeleb2_ID'], keep='first')\

    merged_df.to_csv(cache_path, index=False)
    return ""

def get_celeb_data(path_to_known_ids: str):

    cache_path = "./data/celeb_data_with_wiki.csv"
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    df_with_names = pd.read_csv("./data/vox2_meta.csv")
    id_list = []
    dir_path = Path(path_to_known_ids) 
    with open(dir_path / "metadata.json", "r") as f:
            id_list = list(json.load(f))
    
    df_dev = pd.read_csv("./data/dev.csv")

    # Filter out rows based ids in id_list based on the VoxCeleb2_ID column
    filtered_df = df_with_names[df_with_names['VoxCeleb2_ID'].isin(id_list)]
    filterd_dev = df_dev[df_dev['speaker'].isin(id_list)]
    
    merged_df = filtered_df.merge(
        filterd_dev[['speaker', 'video', 'start_time']],
        left_on='VoxCeleb2_ID',
        right_on='speaker',
        how='left'
    ).drop_duplicates(subset=['VoxCeleb2_ID'], keep='first')\
    .drop(columns=['speaker','Set','VGGFace2_ID', 'Gender'])

 
    # Add Wikipedia info to the dataframe
    wiki_info = merged_df['Name'].apply(get_wiki_info)

    merged_df['image_url'] = wiki_info.apply(lambda x: x['image_url'])
    merged_df['wiki_description'] = wiki_info.apply(lambda x: x['wiki_description'])
    merged_df['wiki_url'] = wiki_info.apply(lambda x: x['wiki_url']) 
    
    # Cache the results
    merged_df.to_csv(cache_path, index=False)

    return merged_df
 