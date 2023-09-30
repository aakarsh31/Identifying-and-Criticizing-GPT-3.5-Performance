This project aims to define OpenAI’s ChatGPT chatbot from a poetic/opinion-based sense of view. The main question we would like to ask here is “Can GPT duplicate the responses of humans in areas of little statistic knowledge and more of human opinion-based domains like sentimental analysis or poetic analysis and classification?”. 


We then proceed further to judge what metrics can an AI model be compared to a real-life human keeping in mind that each human has different opinions. Every human's opinions also change with time, making it harder to judge an AI model accurately. 

We proceed to the implementation of the poetry genre classification using GPT-3.5. We use a model that is better for our purposes as we mentioned earlier. GPT-3.5 davinci-001 is used for the purpose of this experiment. We first use various trial and error, prompt testing iterations to get accustomed to multiple  parameters for the Language model to generate outputs that are accurate and according to our needs. We start with temperature and then proceed with top_P and lastly, we explore some prompts that enhance our required output.

![image](https://github.com/aakarsh31/3-2/assets/89195418/365848a1-0abb-4b30-9aca-38ceec1ee9cc)


The dataset was supplied to 4 different humans who would proceed to classify the poems into categories from the set of all GPT-generated poem classes. We take a final class of outputs by measuring the majority of poem genre/class. In cases where there is no appropriate majority, we get a completely new person for a 5th iteration of the conflicting poems. The person assigns poems to their classes from the set of 4 four classes of the initial 4 people. We acquire a majority in poem classification in all cases with this method. For comparison purposes, we use metrics like confusion matrix, accuracy, precision, and F1 Score. This leads us to the methodology for the calculation of the comparison metrics. 

![image](https://github.com/aakarsh31/3-2/assets/89195418/505584b9-9cf8-405a-a7f3-619906da84ba)
