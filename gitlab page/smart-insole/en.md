### Goal
	Easy to use web platform for analyzing walking data, which is collected from various smart insoles. Users can monitor their walking trends via handy graphs after the data has been uploaded and processed on the server. Additionally we have implemented classifiers for fall detection based on the step data. This can be used to get help for people when the classifier is used in real time.


### Implementation
	We have selected and generated features from the data which highlight if there are balance problems with the user. We also have trained multiple models for the fall detection task with our data that we collected. The predictions from them are combined with ensemble methdos such as bagging. This implementation can be deployed for use on our Django powered website and possibly mobile platforms later.
	

### Future development
	- Extending the platform to mobile for real time classifications or running the classifiers locally on mobile
	- Support for other smart insoles
	- Possibly calling for help automatically if user falls
	- Extension to medical usage