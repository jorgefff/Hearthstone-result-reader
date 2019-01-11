
#include "pch.h"
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

//Global Variables
char* window_name = (char*)"Image processed";


/*
	The list of images to be analyzed
*/
cv::Mat IMAGES[] = {
	cv::imread("resources/1920x1080/druid_v.png"),
	cv::imread("resources/1920x1080/druid_d.png"),
	cv::imread("resources/1920x1080/hunter_d.png"),
	cv::imread("resources/1920x1080/hunter_v.png"),
	cv::imread("resources/1920x1080/mage_d.png"),
	cv::imread("resources/1920x1080/mage_v.png"),
	cv::imread("resources/1920x1080/paladin_d.png"),
	cv::imread("resources/1920x1080/paladin_v.png"),
	cv::imread("resources/1920x1080/priest_d.png"),
	cv::imread("resources/1920x1080/priest_v.png"),
	cv::imread("resources/1920x1080/rogue_d.png"),
	cv::imread("resources/1920x1080/rogue_v.png"),
	cv::imread("resources/1920x1080/shaman_d.png"),
	cv::imread("resources/1920x1080/shaman_v.png"),
	cv::imread("resources/1920x1080/warlock_d.png"),
	cv::imread("resources/1920x1080/warlock_v.png"),
	cv::imread("resources/1920x1080/warrior_d.png"),
	cv::imread("resources/1920x1080/warrior_v.png"),
	cv::imread("resources/1920x1080/warrior.png"),

	cv::imread("resources/1280x720/resized_druid_v.png"),
	cv::imread("resources/1280x720/resized_hunter_v.png"),
	cv::imread("resources/1280x720/resized_mage_v.png"),

	cv::imread("resources/1280x720/druid_v.jpg"),
	cv::imread("resources/1280x720/hunter_v.jpg"),
	cv::imread("resources/1280x720/mage_v.jpg"),
	cv::imread("resources/1280x720/resized_druid_d.png"),
	cv::imread("resources/1280x720/resized_hunter_d.png"),
	cv::imread("resources/1280x720/resized_mage_d.png"),
	cv::imread("resources/1280x720/paladin_v.jpg"),
	cv::imread("resources/1280x720/priest_v.jpg"),
	cv::imread("resources/1280x720/rogue_v.jpg"),
	cv::imread("resources/1280x720/warlock_v.jpg")
};
int const NUM_IMAGES = (int)(sizeof(IMAGES) / sizeof(IMAGES[0]));

/*
	Template of the classes
*/
cv::Mat CLASS_TEMPLATES[] = {
	cv::imread("resources/icons/Druid_canny.jpg"),
	cv::imread("resources/icons/Hunter_canny.jpg"),
	cv::imread("resources/icons/Mage_canny.jpg"),
	cv::imread("resources/icons/Paladin_canny.jpg"),
	cv::imread("resources/icons/Priest_canny.jpg"),
	cv::imread("resources/icons/Rogue_canny.jpg"),
	cv::imread("resources/icons/Warlock_canny.jpg"),
	cv::imread("resources/icons/Warrior_canny.jpg"),
	cv::imread("resources/icons/Shaman_canny.jpg")
};
int const NUM_CLASSES = (int)(sizeof(CLASS_TEMPLATES) / sizeof(CLASS_TEMPLATES[0]));

/*
	Templates for "Victory" and "Defeat"
*/
cv::Mat RESULT_TEMPLATES[] = {
	cv::imread("resources/icons/victory_canny.png"),
	cv::imread("resources/icons/defeat_canny.png")
};
int const NUM_RESULTS = (int)(sizeof(RESULT_TEMPLATES) / sizeof(RESULT_TEMPLATES[0]));


/* 
	Translates the int index to the name of the matching method 
*/
string match_method_name(int method)
{
	switch (method)
	{
		case 0:	return "SQDIFF";
		case 1: return "SQDIFF NORMED";
		case 2:	return "TM CCORR";
		case 3: return "TM CCORR NORMED";
		case 4: return "TM COEFF";
		case 5: return "TM COEFF NORMED";
		default: return "INVALID NUM";
	}
}


/*
	Translates the index to the name of the class 	
*/
string get_class_name (int i)
{
	switch (i)
	{
		case 0:	return "Druid";
		case 1: return "Hunter";
		case 2:	return "Mage";
		case 3: return "Paladin";
		case 4: return "Priest";
		case 5: return "Rogue";
		case 6: return "Warlock";
		case 7: return "Warrior";
		case 8: return "Shaman";
		default: return "INVALID NUM";
	}
}


/*
	Translates the index to the result 	
*/
string victory_or_defeat(int i)
{
	if (i == 0)
		return "Victory";
	
	if (i == 1)
		return "Defeat";

	return "UNKNOWN RESULT";
}


/*
	Apply canny filter to an image
*/
cv::Mat apply_canny(cv::Mat original, int low_threshold = 86, int thresh_ratio = 3)
{
	// Convert the image to grayscale
	cv::Mat image_gray;
	cvtColor(original, image_gray, cv::COLOR_RGB2GRAY);

	// Reduce noise with a 3x3 kernel
	cv::Mat detected_edges;
	cv::blur(image_gray, detected_edges, cv::Size(3, 3));

	// Canny filter
	int kernel_size = 3;
	cv::Canny(detected_edges, detected_edges, low_threshold, low_threshold * thresh_ratio, kernel_size);

	// Using Canny's output as a mask
	cv::Mat img_canny;
	img_canny = cv::Scalar::all(0);
	original.copyTo(img_canny, detected_edges);

	return img_canny;
}


/*
	Returns center of an image
	@param type:0 if class icon, 1 if result
*/
cv::Point get_image_center(cv::Mat image, int type)
{
	// Templates used are rectangles
	// Part of the class icon is cropped because it has curves
	int y_offset;
	
	// The vertical center of the class image in comparison with the full image height
	double placement_ratio;

	if (type == 0)
	{
		y_offset = 20;
		placement_ratio = 0.375;
	}
	else
	{
		y_offset = 0;
		placement_ratio = 0.5805;
	}
	
	cv::Point img_center;
	img_center.x = (int)(image.cols / 2);
	img_center.y = (int)(image.rows * placement_ratio) + y_offset;

	return img_center;
}


/*
	Finds the lowest value in an array
*/
void get_best_match(double worst_case, int *tmpl_matched, double templ_score[], int size)
{
	double best_val = worst_case;
	for (int tmpl_idx = 0; tmpl_idx < size; tmpl_idx++)
	{
		if (templ_score[tmpl_idx] < best_val)
		{
			best_val = templ_score[tmpl_idx];
			*tmpl_matched = tmpl_idx;
		}
	}
}


/*
	Uses template matching to find which class was played
*/
int template_match_class(cv::Mat image)
{	
	double templ_score[NUM_CLASSES];
	
	cv::Mat original_img = image.clone();

	// Validate and correct dimensions
	if (original_img.rows != 720)
	{
		int new_width = (original_img.cols * 720) / original_img.rows;
		cv::resize(original_img, original_img, cv::Size(new_width, 720));
	}

	// Shows the image being analyzed
	cv::imshow(window_name, original_img);
	cv::waitKey(1);

	// The center of the class icon
	cv::Point img_center = get_image_center(original_img, 0);

	// Apply canny filter
	cv::Mat img_canny = apply_canny(original_img);
		
	// Use template matching with all class templates
	for (int tmpl_idx = 0; tmpl_idx < NUM_CLASSES; tmpl_idx++)
	{
		// Loads the current template to be matched
		cv::Mat tmpl = CLASS_TEMPLATES[tmpl_idx];

		// Create the result matrix
		cv::Mat result;
		int result_cols = original_img.cols - tmpl.cols + 1;
		int result_rows = original_img.rows - tmpl.rows + 1;
		result.create(result_rows, result_cols, CV_32FC1);
		
		// Do the matching
		cv::matchTemplate(img_canny, tmpl, result, cv::TM_CCOEFF_NORMED);

		// Localizing the best match with minMaxLoc
		double min_val, match_val;
		cv::Point min_loc, max_loc, match_loc;
		minMaxLoc(result, &min_val, &match_val, &min_loc, &max_loc, cv::Mat());
		match_loc = max_loc;
			
		// Template center
		cv::Point tmpl_center;
		tmpl_center.x = (int) (tmpl.cols / 2 + match_loc.x);
		tmpl_center.y = (int) (tmpl.rows / 2 + match_loc.y);

		// Distance from template center to class icon center
		double distance = sqrt( pow(abs(img_center.x - tmpl_center.x),2) + pow(abs(img_center.y - tmpl_center.y),2) );
		double score = distance - match_val * 80;

		// Info output
		cout << "Template: " << get_class_name(tmpl_idx) << endl;
		cout << "Score: " << score << endl;
		cout << "----------------" << endl;
		
		// Save result
		templ_score[tmpl_idx] = score;
	}

	// Finds the template with best results
	int tmpl_matched;
	get_best_match((double)original_img.cols, &tmpl_matched, templ_score, NUM_CLASSES);

	return tmpl_matched;
}


/*
	Uses template matching to find the match result
*/
int template_match_result(cv::Mat image)
{
	double templ_score[NUM_RESULTS];
	cv::Mat original_img = image.clone();

	// Validate and correct dimensions
	if (original_img.rows != 720)
	{
		int new_width = (original_img.cols * 720) / original_img.rows;
		cv::resize(original_img, original_img, cv::Size(new_width, 720));
	}

	// Shows the image being analyzed
	cv::imshow(window_name, original_img);
	cv::waitKey(1);

	// The center of the duel result icon
	cv::Point img_center = get_image_center(original_img, 1);

	// Apply canny filter
	cv::Mat img_canny = apply_canny(original_img);

	// Use template matching with victory/defeat templates
	for (int tmpl_idx = 0; tmpl_idx < NUM_RESULTS; tmpl_idx++)
	{
		string v_d = victory_or_defeat(tmpl_idx);

		// Create the result matrix
		cv::Mat result;
		cv::Mat tmpl = RESULT_TEMPLATES[tmpl_idx];

		int result_cols = original_img.cols - tmpl.cols + 1;
		int result_rows = original_img.rows - tmpl.rows + 1;
		result.create(result_rows, result_cols, CV_32FC1);

		// Do the matching
		cv::matchTemplate(img_canny, tmpl, result, cv::TM_CCOEFF_NORMED);

		// Localizing the best match with minMaxLoc
		double min_val, match_val;
		cv::Point min_loc, max_loc, match_loc;
		minMaxLoc(result, &min_val, &match_val, &min_loc, &max_loc, cv::Mat());
		match_loc = max_loc;

		// Template center
		cv::Point tmpl_center;
		tmpl_center.x = (int)(tmpl.cols / 2 + match_loc.x);
		tmpl_center.y = (int)(tmpl.rows / 2 + match_loc.y);

		// Distance from template center to class icon center
		double distance = sqrt(pow(abs(img_center.x - tmpl_center.x), 2) + pow(abs(img_center.y - tmpl_center.y), 2));

		// Info output
		cout << "Template: " << v_d << endl;
		cout << "Score: " << distance - match_val * 10 << endl;
		cout << "----------------" << endl;

		// Save result
		templ_score[tmpl_idx] = distance - match_val * 80;
	}

	// Finds the template with best results
	int tmpl_matched;
	get_best_match((double)original_img.cols, &tmpl_matched, templ_score, NUM_RESULTS);
	
	return tmpl_matched;
}



int main(int argc, char** argv)
{
	// Create a window
	cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

	// Matching
	for (int i = 0; i < NUM_IMAGES; i++)
	{
		int class_id = template_match_class(IMAGES[i]);
		int result_id = template_match_result(IMAGES[i]);

		string class_name = get_class_name(class_id);
		string vic_def = victory_or_defeat(result_id);
		
		cout << "######################" << endl;
		cout << "Class: " << class_name << endl;
		cout << "Duel result: " << vic_def << endl;
		cout << "######################" << endl;;
		cv::waitKey(0);
	}

	cout << "\nAll images were processed" << endl;
	cv::waitKey(0);

	return 0;
}