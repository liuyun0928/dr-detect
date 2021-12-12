# include <diabetic_retinopathy.h>

int DiabeticRetinopathy::Train(std::string path) {

}

int DiabeticRetinopathy::convertLabelToLevel(std::string label) {
    if(label == "No DR")
        return 0;
    else if (label == "Mild")
        return 1;
    else if (label == "Moderate")
        return 2;
    else if (label == "Severe")
        return 3;
    else if (label == "Proliferative DR")
        return 4;
    else
        return 5;
}

std::string DiabeticRetinopathy::convertLevelToClass(int level) {
    switch (level)
    {
    case 0:
        return "No DR";
    case 1:
        return "Mild";
    case 2:
        return "Moderate";
    case 3:
        return "Severe";
    case 4:
        return "Proliferative DR";
    default:
        return ;
    }
}