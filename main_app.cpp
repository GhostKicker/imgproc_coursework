#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace xfeatures2d;

string dir_path = "C:\\ads\\";

//share of how much ads were found
int num_discovered = 0;
int num_ads = 0;

//share of how many wrong maatches there were
int num_matches = 0;
int wrong_matches = 0;

const double pi = acos(-1.0);
const double P = 0.6;
const double K = 3;

cv::Ptr<cv::SIFT> detector = cv::SIFT::create(0, 3, 0.04, 10, 1);
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
void detectKeyPoints(vector<Mat>& vec, vector<vector<KeyPoint>>& kpts, vector<Mat>& descr) {
    for (int i = 0; i < vec.size(); ++i) {
        detector->detectAndCompute(vec[i], noArray(), kpts[i], descr[i]);
        drawKeypoints(vec[i], kpts[i], vec[i]);
        cout << i << "-th detected" << endl;
    }
}


int main(int argc, char** argv)
{
    vector<string> names_pages;
    vector<string> names_mdpages;
    vector<string> names_elements;
    vector<string> names_rects;

    glob(dir_path + "pages\\", names_pages);
    glob(dir_path + "elements\\", names_elements);
    glob(dir_path + "md_pages\\", names_mdpages);
    glob(dir_path + "markdown\\", names_rects);



    vector<Mat> pages(names_pages.size());
    vector<Mat> sources(names_pages.size());
    vector<Mat> elements(names_elements.size());
    vector<Mat> md_pages(names_mdpages.size());
    vector<vector<Rect>> rects(names_rects.size());
    vector<vector<bool>> used(names_rects.size());


    for (int i = 0; i < pages.size(); ++i) pages[i] = imread(names_pages[i], 0);
    for (int i = 0; i < pages.size(); ++i) sources[i] = imread(names_pages[i]);
    for (int i = 0; i < elements.size(); ++i) elements[i] = imread(names_elements[i], 0);
    for (int i = 0; i < md_pages.size(); ++i) md_pages[i] = imread(names_mdpages[i]);
    for (int i = 0; i < rects.size(); ++i) {
        ifstream ifstr(names_rects[i]);
        Rect r;
        int n;
        ifstr >> n;
        num_ads += n;
        for (int j = 0; j < n; ++j) {
            rects[i].push_back(Rect());
            used[i].push_back(false);
            ifstr >> rects[i].back().x;
            ifstr >> rects[i].back().y;
            ifstr >> rects[i].back().width;
            ifstr >> rects[i].back().height;
        }
    }

    vector<vector<KeyPoint>> keypoints_pages(pages.size()), keypoints_elements(elements.size());
    vector<Mat> descriptors_pages(pages.size()), descriptors_elements(elements.size());

    detectKeyPoints(pages, keypoints_pages, descriptors_pages);
    cout << "pages done!" << endl;
    detectKeyPoints(elements, keypoints_elements, descriptors_elements);
    cout << "elements done!" << endl;


    //imwrite("pageeee.PNG", pages[4]);
    //imwrite("elemmmm.PNG", elements[4]);

    int curind = 0;

    for (int curpage = 0; curpage < pages.size(); ++curpage) {


        for (int curelem = 0; curelem < elements.size(); ++curelem) {
            std::vector< std::vector<DMatch>> knn_matches;
            matcher->knnMatch(descriptors_elements[curelem], descriptors_pages[curpage], knn_matches, 2);
            const float ratio_thresh = 0.7f;
            std::vector<DMatch> good_matches;
            vector<Point2f> good_pts_elem;
            vector<Point2f> good_pts_page;
            Point2f center_elem(0, 0);
            Point2f center_page(0, 0);
            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                    good_matches.push_back(knn_matches[i][0]);
                    auto pt_elem = keypoints_elements[curelem][knn_matches[i][0].queryIdx].pt;
                    auto pt_page = keypoints_pages[curpage][knn_matches[i][0].trainIdx].pt;
                    good_pts_elem.push_back(pt_elem);
                    good_pts_page.push_back(pt_page);
                    center_elem += pt_elem;
                    center_page += pt_page;
                }
            }

            if (good_pts_page.size() == 0) continue;
            if (good_pts_elem.size() == 0) continue;
            center_page /= float(good_pts_page.size());
            center_elem /= float(good_pts_elem.size());


            Mat img_matches;
            Mat& pic1 = elements[curelem];
            Mat& pic2 = pages[curpage];
            auto& kp1 = keypoints_elements[curelem];
            auto& kp2 = keypoints_pages[curpage];
            drawMatches(pic1, kp1, pic2, kp2, good_matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            //imshow("Good Matches", img_matches);
            //waitKey();



            bool ok = true;
            if (good_matches.size() < keypoints_elements[curelem].size() * P || good_matches.size() == 0) continue;
            for (int i = 0; i < good_pts_page.size(); ++i) {
                if (K * norm(good_pts_elem[i] - center_elem) < norm(good_pts_page[i] - center_page))
                    ok = false;
                if (!ok) break;
            }
            for (int i = 0; i < good_pts_page.size(); ++i) {
                auto vec1 = good_pts_elem[i] - center_elem;
                auto vec2 = good_pts_page[i] - center_page;
                double cross = vec1.cross(vec2);
                double dot = vec1.dot(vec2);
                if (abs(atan2(cross, dot)) > pi / 2)
                    ok = false;
                if (!ok) break;
            }
            if (!ok) cout << "bad" << endl;
            if (!ok) continue;
            cout << "good" << endl;

            circle(sources[curpage], center_page, 3, Scalar(255, 0, 255), 4);

            num_matches++;
            int wr = 1;
            for (int i = 0; i < rects[curpage].size(); ++i) {
                auto& it = rects[curpage][i];
                if (it.contains(center_page)) {
                    wr = 0;
                    used[curpage][i] = 1;
                }
            }

            wrong_matches += wr;


            //imwrite(dir_path + "result\\" + to_string(curind++) + ".png", img_matches);
        }
        imwrite(dir_path + "result\\" + to_string(curind++) + ".png", sources[curpage]);
    }

    for (auto& it : used) {
        for (auto& itt : it) num_discovered += itt;
    }

    ofstream gogo(dir_path + "statistics.txt");
    gogo << "Share of what was discovered: " << num_discovered << "/" << num_ads << endl;
    gogo << "Share of wrong matches: " << wrong_matches << "/" << num_matches << endl;
    //waitKey(0);


    return 0;
}