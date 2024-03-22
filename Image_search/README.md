
# Flask Based WebApp connecting CLIP and ChromaDb for efficient Image Search

The project is a Flask-based web application deployed using Docker. It utilizes the CLIP (Contrastive Language-Image Pretraining) model for text and image embeddings and ChromaDB, a database server, for storing and querying data. Users can add data to the system, which includes uploading images and providing metadata. The system then extracts embeddings for the images using the CLIP model and stores the data in the ChromaDB collection. Users can also query the database using text or images to retrieve relevant results.

##  Code Structure
The project consists of several components:

+ Script.py: This file contains the CLIPModelHelper, DataHelper, and ChromaDBManager classes. These classes provide functionality for interacting with the CLIP model, converting data types, and managing the ChromaDB collection, respectively.

+ App.py: This is the main Flask application file. It handles user authentication, data uploading, and querying. It also initializes the CLIP model helper and ChromaDB manager.

+ Dockerfile: This file is used to build the Docker image for the application. It specifies the base image, sets the working directory, copies the project files, installs dependencies from requirements.txt, and specifies the command to run the application.

+ Docker-compose.yml: This file defines the services for the application and ChromaDB server. It specifies the build context for the application, exposes ports, sets environment variables, and defines volumes for data persistence.

## Setting Up and Running the Application
To set up and run the application, follow these steps:

+ Clone the project repository to your local machine.
+ Navigate to the project directory in the terminal.
+ Run
  ``` docker-compose up --build ``` 
to build the Docker image and start the services.
+ Access the application at http://localhost:4000 in your web browser.



## Docker Compose Configuration Documentation
  ### Services
*App Service*
+ Description: This service defines the main application container.
+ Build: The service is built using the Dockerfile located in the current directory (.).

+ Ports: The container's port 4000 is mapped to the host's port 4000, allowing access to the application.

+ Dependencies: This service depends on the chromadb-server service.
+ Environment Variables:
  CHROMADB_HOST: The hostname of the ChromaDB server, set to chromadb-server.
+ Volumes:
  ``` ./static:/app/static ``` : Mounts the static directory from the host to the /app/static directory in the container, allowing the application to access static files.
chromadb-server Service
+ Description: This service defines the ChromaDB server container.
+ Image: The service uses the chromadb/chroma image from Docker Hub.
+ Ports: The container's port 8000 is mapped to the host's port 8000, allowing access to the ChromaDB server.
### Docker Compose Version
Version: 3.8



## Using the Application
+ Login: Users can log in using their username and password. Admins can access additional features after logging in.
+ Signup: New users can sign up for an account by providing a username and password.
+ Add Data: Users can add data to the system by uploading a CSV file containing metadata and a ZIP file containing images. The system extracts embeddings for the images using the CLIP model and stores the data in the ChromaDB collection.
+ Query Data: Users can query the database using text or images to retrieve relevant results. The system uses the CLIP model to extract embeddings for the query and retrieves similar items from the database.
+ Logout: Users can log out of the application to end their session.




# Deep Down in Script.py file
## 1. CLIP Model Helper
### CLIPModelHelper class
### Description: 
Provides helper functions for working with the CLIP model, including initializing the model, getting embeddings for text and images, and working with DataFrame inputs.
### Initialization:
+ model_ID: Identifier for the pre-trained CLIP model
+ device: Device to use ('cpu' or 'cuda').

### Methods:
+ get_model_info(self, model_ID, device): <br/>

    Input: model_ID (str), device (str). <br/>
    Output: Tuple containing the CLIP model, processor, and tokenizer.<br/>
    Description: Initializes and returns the CLIP model, processor, and tokenizer.<br/>
+ get_single_text_embedding(self, text, device): <br/>

   Input: text (str), device (str). <br/>
   Output: Numpy array containing the CLIP embedding for the input text.<br/>
 Description: Gets the CLIP embedding for a single text input.<br/>
 
+ get_all_text_embeddings(self, df, text_col, device): <br/>

  Input: df (DataFrame), text_col (str), device (str). <br/>
  Output: DataFrame with an additional column containing CLIP embeddings for each text input. <br/>
  Description: Gets CLIP embeddings for all text inputs in a DataFrame. <br/>

+ get_single_image_embedding(self, my_image, device): <br/>

  Input: my_image (numpy array), device (str). <br/>
  Output: Numpy array containing the CLIP embedding for the input image. <br/>
  Description: Gets the CLIP embedding for a single image input. <br/>

+ get_all_images_embedding(self, df, img_column, device): <br/>

  Input: df (DataFrame), img_column (str), device (str). <br/>
  Output: DataFrame with an additional column containing CLIP embeddings for each image input. <br/>
  Description: Gets CLIP embeddings for all image inputs in a DataFrame. <br/>

## 2. Data Helper
### DataHelper class
### Description: Provides helper functions for converting data types, particularly converting string representations to NumPy arrays.
### Methods:
+ convert_string_to_array(input_string):<br/>

  Input: input_string (str).<br/>
  Output: Formatted string representing the input as a NumPy array.<br/>
  Description: Converts a string representation to a NumPy array.<br/>
  
+ convert_to_numpy_array(df):<br/>

  Input: df (DataFrame).<br/>
  Output: DataFrame with columns converted to NumPy arrays.<br/>
  Description: Converts DataFrame columns from string representations to NumPy arrays.<br/>

## 3. ChromaDB Manager
### ChromaDBManager class
### Description: 
Provides functions for managing a ChromaDB collection, including initializing the collection, adding and updating data, and querying the collection.
### Methods:
+ __init__(self, host, port)<br/>

  Input: host (str), port (int).<br/>
  Description: Initializes the ChromaDB HTTP client and collection.<br/>
  
+ initialize_collection(self, collection_name)<br/>

  Input: collection_name (str).<br/>
  Description: Initializes a collection in ChromaDB with the given name.<br/>
  
+ add_data_to_collection(self, images_df, df, metadatas, img_embedding_column='flattened_embeddings', id_column='id')<br/>

  Input: images_df (DataFrame), df (DataFrame), metadatas (list), img_embedding_column (str), id_column (str).<br/>
  Description: Adds data to the ChromaDB collection.<br/>
  
+ change_metadata_from_collection(self, text_df, images_df, df_change, metadatas_change, text_column='text', img_embedding_column='flattened_embeddings', id_column='id')<br/>

  Input: text_df (DataFrame), images_df (DataFrame), df_change (list), metadatas_change (list), text_column (str),   <br/>    
  img_embedding_column (str), id_column (str).<br/>
  Description: Changes metadata in the ChromaDB collection for specified IDs.<br/>

+ delete_from_collection(self, df_del)<br/>

  Input: df_del (list).<br/>
  Description: Deletes entries from the ChromaDB collection.<br/>

+ get_from_collection(self, df_get)<br/>

  Input: df_get (list).<br/>
  Output: Retrieved entries from the collection.<br/>
  Description: Retrieves entries from the ChromaDB collection based on specified IDs.<br/>

+ query_collection(self, query_emb, n_results)<br/>

  Input: query_emb (list), n_results (int).<br/>
  Output: Results of the query.<br/>
  Description: Queries the ChromaDB collection based on specified query embeddings.<br/>
