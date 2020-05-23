#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

string dir_path = "C:\\ads\\";
Mat pic_source;
Mat pic;
vector<Rect> rects;
bool previous = false;

void mouse_callback(int  event, int  x, int  y, int  flag, void* param) {
    static int curx, cury;
    if (event == EVENT_LBUTTONDOWN) {
        if (previous == false) {
            curx = x;
            cury = y;
            previous = true;
        }
        else {
            rectangle(pic, Rect(Point(curx, cury), Point(x, y)), Scalar(0, 0, 255), 2);
            previous = false;
            rects.push_back(Rect(Point(curx, cury), Point(x, y)));
        }
        cout << "(" << x << ", " << y << ")" << endl;
        imshow("example", pic);
    }
}

void writeRects(int i) {
    string name = dir_path + "markdown\\" + to_string(i) + ".txt";
    ofstream fstr(name);
    fstr << rects.size() << endl;
    for (auto& it : rects) {
        fstr << it.x << " " << it.y << " " << it.width << " " << it.height << endl;
        cout << it.x << " " << it.y << " " << it.width << " " << it.height << endl;
    }
}

void writeMarkedDown(int i) {
    string name = dir_path + "md_pages\\" + to_string(i) + ".PNG";
    imwrite(name, pic);
}

void writeAds() {
    static int i = 0;
    for (auto& r : rects) {
        string name = dir_path + "elements\\" + to_string(i++) + ".PNG";
        imwrite(name, pic_source(r));
    }
}

int main()
{
    string dir = dir_path + "pages\\";
    vector<string> files;
    glob(dir, files);
    for (int i = 0; i < files.size(); ++i) {
        rects.clear();
        previous = false;
        string name = files[i];

        cout << name << endl;
        pic = imread(name);
        pic_source = imread(name);

        namedWindow("example");
        setMouseCallback("example", mouse_callback);
        imshow("example", pic);
        waitKey();

        for (auto& r : rects) cout << r << endl;
        writeRects(i);
        writeMarkedDown(i);
        writeAds();
    }


    return 0;
}