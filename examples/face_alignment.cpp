// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    This face detector is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset.  

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


#include <dlib/image_processing/frontal_face_detector.h>
// #include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
// #include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shapesmall.dat") >> sp;
    // serialize("shapesmall.dat") << sp;

    array2d<rgb_pixel> img;
    load_image(img, argv[1]);
    // pyramid_up(img);

    std::vector<rect_detection> d;    //rectangle d.rect; double d.detection_confidence
    detector(img,d,-0.35);
    for(int i=0;i<d.size();i++)
    {
        // cout << d[i].rect << " s = " << d[i].detection_confidence << endl;
        // draw_rectangle(img,d[i].rect,rgb_pixel(255,0,0));
    }
    // save_jpeg(img,"test-rect.jpg",95);

    // img = sub_image(img,d[0].rect); 
    // extract_image_chip(img,d[0].rect,img);
    // save_jpeg(img,"test-ex-2.jpg",95);
    // full_object_detection shape = sp(img, rectangle(0,0,d[0].rect.width(),d[0].rect.height()));
    full_object_detection shape = sp(img, d[0].rect);


    std::vector<dlib::vector<double,2> > f, t;
    //OUTER_EYES_AND_NOSE = [36, 45, 33]
    // draw_solid_circle(img,shape.part(36),2,255);
    // draw_solid_circle(img,shape.part(45),2,255);
    // draw_solid_circle(img,shape.part(33),2,255);
    // cout << shape.part(36) << endl;
    // cout << shape.part(45) << endl;
    // cout << shape.part(33) << endl;
    t.push_back(shape.part(36));
    t.push_back(shape.part(45));
    t.push_back(shape.part(33));
    // f.push_back(dlib::vector<double,2>(18.50,16.75)); //18.64,16.25
    // f.push_back(dlib::vector<double,2>(76.50,15.70)); //75.73,15.18
    // f.push_back(dlib::vector<double,2>(47.52,48.20)); //47.52,49.39
    f.push_back(dlib::vector<double,2>(18.64,16.25)); //18.64,16.25
    f.push_back(dlib::vector<double,2>(75.73,15.18)); //75.73,15.18
    f.push_back(dlib::vector<double,2>(47.52,49.39)); //47.52,49.39

    array2d<rgb_pixel> align;
    set_image_size(align, 96, 96);
    transform_image(img,align,interpolate_bilinear(),find_affine_transform(f, t));
    save_jpeg(align,"test-align.jpg",95);


    // // image_window win, win_faces;
    // // Loop over all the images provided on the command line.
    // for (int i = 2; i < argc; ++i)
    // {
    //     cout << "processing image " << argv[i] << endl;
    //     array2d<rgb_pixel> img;
    //     load_image(img, argv[i]);
    //     // Make the image larger so we can detect small faces.
    //     pyramid_up(img);

    //     // Now tell the face detector to give us a list of bounding boxes
    //     // around all the faces in the image.
    //     std::vector<rectangle> dets = detector(img);
    //     cout << "Number of faces detected: " << dets.size() << endl;

    //     // Now we will go ask the shape_predictor to tell us the pose of
    //     // each face we detected.
    //     std::vector<full_object_detection> shapes;
    //     for (unsigned long j = 0; j < dets.size(); ++j)
    //     {
    //         full_object_detection shape = sp(img, dets[j]);
    //         cout << "number of parts: "<< shape.num_parts() << endl;
    //         cout << "pixel position of first part:  " << shape.part(0) << endl;
    //         cout << "pixel position of second part: " << shape.part(1) << endl;
    //         // You get the idea, you can get all the face part locations if
    //         // you want them.  Here we just store them in shapes so we can
    //         // put them on the screen.
    //         shapes.push_back(shape);

    //         for(int i=0;i<68;i++)
    //         {
    //             draw_solid_circle(img,shape.part(i),2,255);
    //         }
    //     }

    //     // Now let's view our face poses on the screen.
    //     // win.clear_overlay();
    //     // win.set_image(img);
    //     // win.add_overlay(render_face_detections(shapes));

    //     // We can also extract copies of each face that are cropped, rotated upright,
    //     // and scaled to a standard size as shown here:
    //     dlib::array<array2d<rgb_pixel> > face_chips;
    //     extract_image_chips(img, get_face_chip_details(shapes), face_chips);
    //     // win_faces.set_image(tile_images(face_chips));
    //     save_jpeg(tile_images(face_chips),"aa.jpg",95);

    //     // array2d<rgb_pixel> oldimage = img;
    //     array2d<rgb_pixel> align;
    //     set_image_size(align, 1000, 1000);

    //     std::vector<dlib::vector<double,2> > f, t;
    //     f.push_back(point(200,200));
    //     f.push_back(point(300,300));
    //     f.push_back(point(100,300));
    //     t.push_back(point(200,200));
    //     t.push_back(point(300,300));
    //     t.push_back(point(120,320));

    //     transform_image(img,align,interpolate_bilinear(),
    //         find_affine_transform(f, t));
    //     // rotate_image(img,align,90,interpolate_nearest_neighbor());
    //     save_jpeg(align,"aa-align.jpg",95);
        

    //     // cout << "Hit enter to process the next image..." << endl;
    //     // cin.get();
    // }

}

// ----------------------------------------------------------------------------------------

