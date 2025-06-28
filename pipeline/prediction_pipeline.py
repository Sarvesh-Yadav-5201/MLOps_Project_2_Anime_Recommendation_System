## ................ PREDICTION PIPELINE ................ ## 

# importing libraries
from config.paths_config import *
from utils.helper import *


## ................ PREDICTION PIPELINE ................ ##
## Function for HYBRID recommendation system:

def get_hybrid_recommendation (user_id,
                               n = 5,
                               user_weight = 0.5):
    
    '''
    Function to get hybrid recommendations for a user based on user-based and content-based recommendations.

    Parameters:
        user_id (int): The ID of the user for whom recommendations are to be generated.
        n (int): The number of recommendations to return. Default is 10.
        user_weight (float): The weight given to user-based recommendations. Default is 0.5.

    Returns:
        DataFrame: A DataFrame containing the top n hybrid recommendations for the user, with their scores.
    Note:
        The function combines user-based recommendations and content-based recommendations.
    
    '''

    try:
        ## ...............Get the user-based recommendation data:.................................##
        user_reco_anime=  get_user_recomendations(user_id, n = n)

        # Extract Anime_ids from the recommendations
        anime_ids_list = user_reco_anime['anime_id'].values.tolist()
        anime_name_list = user_reco_anime['NAME'].values.tolist()

       
        ## ...............Get the content-based recommendation data:.................................##

        content_reco_anime_list  = [] # Store all the content-based recommendations for each anime_id

        for (anime_id, name) in zip(anime_ids_list, anime_name_list):
            # Get the similar recommendations for each anime_id
            similar_animes = get_similar_recommendations(anime_id, n=n)

            if similar_animes is not None and not similar_animes.empty:
                content_reco_anime_list.extend(similar_animes["NAME"].tolist())
            else:
                print(f"No similar anime found for anime : {name}")


        ## ......................... Combine the user-based and content-based recommendations:..................................##
        
        # Define the content weights : 
        content_weight = 1 - user_weight

        combined_scores = {}

        for anime in anime_name_list:
            combined_scores[anime] = combined_scores.get(anime,0) + user_weight

        for anime in content_reco_anime_list:
            combined_scores[anime] = combined_scores.get(anime,0) + content_weight 

        ## Sort the combined scores in descending order
        sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        ## Get Top n Recommendations:
        sorted_combined_scores = sorted_combined_scores[:n]

        ## Create a DataFrame from the sorted combined scores
        recommendations_df = pd.DataFrame(sorted_combined_scores, columns=['Anime Name', 'Score'])

        return recommendations_df['Anime Name'].tolist()
    

    except Exception as e:
        print('ERROR: Hybrid recommendation system not implemented yet.', e)
        return None
    


if __name__ == "__main__":
    ## Test the hybrid recommendation system:
    print('Testing the hybrid recommendation system...')
    print('This may take a few minutes...')

    ## Get hybrid recommendations:
    data = get_hybrid_recommendation(2,  
                                n = 5,
                                user_weight = 0.5)


    print ('Top Recommendations for the user:')
    print(data)
