## .......... HELPER FUNCTION FOR RECOMMENDATION SYSTEMS............ ##
import numpy as np
import pandas as pd
import joblib
from config.paths_config import *


######################################################################

## 1. GET ANIME FRAME

def get_anime_frame(anime, path_anime_df = ANIME_DF):

    ## Get the anime DataFrame from the specified path
    try:
        df = pd.read_csv(path_anime_df)
    except FileNotFoundError:
        print(f"File not found at {path_anime_df}")
        return None

    # if the instance is integer:
    if isinstance(anime, int):
        # Check if the anime_id exists in the DataFrame
        if anime in df['anime_id'].values:
            # Return the row corresponding to the anime_id
            return df[df['anime_id'] == anime]
        else:
            return None
        
    # if the instance is string:
    elif isinstance(anime, str):
        # Check if the anime name exists in the DataFrame
        if anime in df['eng_version'].values:
            # Return the row corresponding to the anime name
            return df[df['eng_version'] == anime]
        else:
            return None
    else:
        return None
    
   

## 2. GET SYNOPSIS DATA

def get_synopsys(anime, path_synop_df = SYNOPSIS_DF):
    ## Get the anime synopsis DataFrame from the specified path
    try:
        df = pd.read_csv(path_synop_df)
    except FileNotFoundError:
        print(f"File not found at {path_synop_df}")
        return None
    
    # if the instance is integer:
    if isinstance(anime, int):
        # Check if the anime_id exists in the DataFrame
        if anime in df['anime_id'].values:
            # Return the row corresponding to the anime_id
            return df[df['anime_id'] == anime]['sypnopsis'].values[0]
        else:
            return None
        
    # if the instance is string:
    elif isinstance(anime, str):
        # Check if the anime name exists in the DataFrame
        if anime in df['Name'].values:
            # Return the row corresponding to the anime name
            return df[df['Name'] == anime]['sypnopsis'].values[0]
        else:
            return None
    else:
        return None 

## 3. CONTENT RECOMMENDATION FUNCTION

def get_similar_recommendations(name, 
                                anime_weights_path = ANIME_WEIGHTS_FILE_PATH , 
                                anime2anime_encoded_path = ANIME2ANIME_ENCODED , 
                                anime2anime_decoded_path = ANIME2ANIME_DECODED , 
                                anime_df_path = ANIME_DF , 
                                synopsys_df_path = SYNOPSIS_DF , 
                                n = 10,  
                                return_dist = False , 
                                neg = False):

    try:
        ## LOAD THE DATA FROM THE SPECIFIED PATHS
        try:
            # anime weights
            anime_weights = joblib.load(anime_weights_path)

            # encoded and decoded anime mappings
            anime2anime_encoded = joblib.load(anime2anime_encoded_path)
            anime2anime_decoded = joblib.load(anime2anime_decoded_path)


        except Exception as e:   
            print(f"File not found at", e)
            return None
        
        
        # get the index of the anime 
        index = get_anime_frame(name).anime_id.values[0]

        # get the encoded value of the anime
        encoded_index = anime2anime_encoded[index]

        # get the weights of the anime
        dists = np.dot( anime_weights, anime_weights[encoded_index] )

        # sorting the distances in descending order
        sorted_dists= np.argsort(dists)
        
        # While recommendation include the anime itself
        n +=1 

        # if neg is True, then we will return the negative recommendations
        if neg:
            closest  = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

        similarity_arr= []

        for close in closest:
            # Get the anime ID from the encoded value
            anime_id = anime2anime_decoded[close]

            # Get the synopsis of the anime
            synopsis = get_synopsys(anime_id)

            # Get Anime frame:
            anime_frame = get_anime_frame(anime_id)

            if return_dist:
                return dists , closest

            ## Extract all the relevant details from the anime frame
            anime_details = {
                'anime_id': anime_frame['anime_id'].values[0],
                'NAME': anime_frame['eng_version'].values[0],
                'Score': anime_frame['Score'].values[0],
                'Genres': anime_frame['Genres'].values[0],
                'Similarity': dists[close],
                'Episodes': anime_frame['Episodes'].values[0],
                'Type': anime_frame['Type'].values[0],
                'Premiered': anime_frame['Premiered'].values[0],
                'Members': anime_frame['Members'].values[0],
                'Synopsis': synopsis,
            }

            ## Append the anime details to the similarity array
            similarity_arr.append(anime_details)


        # Create a DataFrame from the similarity array
        recommendations_df = pd.DataFrame(similarity_arr).sort_values(by='Similarity', ascending=False)

        # return recommendations_df[recommendations_df.anime_id != index].drop(['anime_id'] , axis = 1).reset_index(drop = True)
        return recommendations_df.drop(['anime_id'], axis = 1).reset_index(drop = True)


    except Exception as e:
        print ('ERROR : Anime not found in the database. Please check the anime name or ID.', e)
        return None


## 4. FIND SIMILAR USERS FUNCTION

def find_similar_users(user_id, 
                       user_weights_path = USER_WEIGHTS_FILE_PATH, 
                       user2user_encoded_path = USER2USER_ENCODED, 
                       user2user_decoded_path = USER2USER_DECODED, 
                       n = 10, 
                       return_dist = False, 
                       negative = False ):

    try:
        ## LOAD THE DATA FROM THE SPECIFIED PATHS
        try:
            # user weights
            user_weights = joblib.load(user_weights_path)

            # encoded and decoded anime mappings
            user2user_encoded = joblib.load(user2user_encoded_path)
            user2user_decoded = joblib.load(user2user_decoded_path)

        except Exception as e:
            print(f"File not found at", e)
            return None

        ## Check if the user_id exists in the user2user_encoded mapping 

        if user_id not in user2user_encoded:
            print(f"ERROR: User ID {user_id} not found in the database.")
            return None
        
        # Get the encoded value of the user
        encoded_user_id = user2user_encoded[user_id]

        # Calculate the distances between the user weights and the specified user
        dists = np.dot(user_weights, user_weights[encoded_user_id])

        # Sort the distances in descending order
        sorted_dists = np.argsort(dists)

        # While recommendation include the user itself
        n += 1
        # If negative is True, then we will return the negative recommendations
        if negative:
            closest = sorted_dists[:n]
        else:
            closest = sorted_dists[-n:]

                
        similarity_arr = []

        for close in closest:
            # Get the user ID from the encoded value
            user = user2user_decoded[close]

            # Extract relevant details for the user
            user_details = {
                'similar_user_id': user,
                'Similarity': dists[close],
            }

            # Append the user details to the similarity array
            similarity_arr.append(user_details)

        # Create a DataFrame from the similarity array
        similar_users = pd.DataFrame(similarity_arr).sort_values(by='Similarity', ascending=False)

        # removing the user itself from the recommendations
        similar_users = similar_users[similar_users.similar_user_id != user_id].reset_index(drop=True)

        if return_dist:
            return similar_users, dists, closest

        return similar_users  # Return the top n similar users



    except:
        print ('ERROR : User not found in the database. Please check the user ID.')
        return None
    

## 5. GET USER PREFERENCES FUNCTION
def get_user_preferences(user_id, 
                         ratings_df_path = RATINGS_DF,
                         anime_df_path = ANIME_DF, 
                        ):
    
    ## Loding the ratings DataFrame from the specified path
    try:
        ratings_df = pd.read_csv(ratings_df_path)
    except FileNotFoundError:
        print(f"File not found at {ratings_df_path}")
        return None
    
    ## Loding the anime DataFrame from the specified path
    try:
        anime_df = pd.read_csv(anime_df_path)
    except FileNotFoundError:
        print(f"File not found at {anime_df_path}")
        return None

    ## Get all the ratings given by the user for animes
    anime_watched_user = ratings_df[ratings_df['user_id'] == user_id]

    ## If the user has not watched any anime, return None
    if anime_watched_user.empty:
        print(f"User ID {user_id} has not watched any anime.")
        return None
    
    ## Get top 10% animes rated by the user:
    n = 10
    percentile_ = np.percentile(anime_watched_user['rating'], 100 - n)
    anime_watched_user = anime_watched_user[anime_watched_user['rating'] >= percentile_].sort_values(by = 'rating', ascending = False) ## Filtering high rated anime

    ## Getting anime ids from the sorted dataframe
    top_anime_user = anime_watched_user['anime_id'].tolist()

    ## Getting anime rows from top animes
    top_anime_user = anime_df[anime_df['anime_id'].isin(top_anime_user)][['eng_version' , 'Genres']]

    return top_anime_user


## 6. GET USER RECOMMENDATIONS FUNCTION

def get_user_recomendations(user_id,
                            n=5):
    try:
        ## Get similar users:
        similar_users = find_similar_users(user_id)

        ## Get user preferences:
        user_preferences = get_user_preferences(user_id)

        ## Empty list to hold recommendations
        recommendation_animes = []
        anime_list = []

        ## Loop through each similar user
        for user_id in similar_users.similar_user_id.values:
            ## Extracting preferences of all the users 
            pref_list = get_user_preferences(int(user_id))

            ## Filter all animes which are already rated by current user
            pref_list = pref_list[~pref_list.eng_version.isin(user_preferences.eng_version.values)]

            ## if the pref list is not empty then append the list to anime_list
            if not pref_list.empty:
                anime_list.append(pref_list)


        ## Concatinate all the animes together:
        if not isinstance(anime_list, list) or not all(isinstance(df, pd.DataFrame) for df in anime_list):
            raise ValueError("Input must be a list of pandas DataFrames.")
    
        anime_user_df =  pd.concat(anime_list, ignore_index=False)

        # get top n anime recommendation: 
        for (anime_id, n_liked) in anime_user_df.eng_version.value_counts()[:n].items():
                # Get anime frame:
                anime_frame = get_anime_frame(anime_id)
            
                anime_details = {
                'anime_id': anime_frame['anime_id'].values[0],
                'NAME': anime_frame['eng_version'].values[0],
                'Score': anime_frame['Score'].values[0],
                'Users liked' : n_liked,
                'Genres': anime_frame['Genres'].values[0],
                'Episodes': anime_frame['Episodes'].values[0],
                'Type': anime_frame['Type'].values[0],
                'Premiered': anime_frame['Premiered'].values[0],
                'Members': anime_frame['Members'].values[0],
                'Synopsis': get_synopsys(int(anime_frame['anime_id'].values[0]))
                }

                ## Appending this data to recommendation_animes
                recommendation_animes.append(anime_details)
        
        return pd.DataFrame(recommendation_animes)

    except Exception as e:
        print('ERROR: User-based recommendations not implemented yet.', e)
        return None
    

