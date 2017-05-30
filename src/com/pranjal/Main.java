package com.pranjal;

import libsvm.*;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Vector;

public class Main {

    static class Point {

        public double x, y;

        public Point (String data) {
            data = data.trim();
            String[] values = data.split("\\s+");
            this.x = Double.parseDouble(values[0]);
            this.y = Double.parseDouble(values[1]);
        }

        public Point (double x, double y) {
            this.x = x;
            this.y = y;
        }

    }

    static class FeatureSet {

        public int expression;

        public Point mean;

        public Point[] points;

        public Point[] shiftedPoints;

        public double[] distances;

        public double[] angles;

        public int angleNose;

        public FeatureSet (String data, String type) {

            switch (type) {
                case "anger" :
                    expression = 6;
                    break;
                case "disgust" :
                    expression = 7;
                    break;
                case "happy" :
                    expression = 8;
                    break;
                case "neutral" :
                    expression = 5;
                    break;
                case "surprise" :
                    expression = 9;
                    break;
            }

            points = new Point[68];
            shiftedPoints = new Point[68];
            distances = new double[68];
            angles = new double[68];
            String[] pointDataArray = data.split("\n");
            for (int i = 0; i < 68; i++) {
                points[i] = new Point(pointDataArray[i]);
            }
            modifyPoints();
        }

        public void setMean () {
            double sumX = 0, sumY = 0;
            for (Point p : points) {
                sumX += p.x;
                sumY += p.y;
            }
            mean = new Point(sumX/68, sumY/68);
        }

        public void modifyPoints () {
            setMean();
            for (int i = 0; i < 68; i++) {
                shiftedPoints[i] = new Point(points[i].x - mean.x, points[i].y - mean.y);
            }
            if (points[26].x == points[29].x) {
                angleNose = 0;
            } else {
                angleNose = (int) (Math.atan((points[26].y - points[29].y)/(points[26].x - points[29].x))*180/Math.PI);
            }
            if (angleNose < 0) {
                angleNose += 90;
            } else {
                angleNose -= 90;
            }
            for (int i = 0; i < 68; i++) {
                distances[i] = Math.sqrt(shiftedPoints[i].x*shiftedPoints[i].x + shiftedPoints[i].y*shiftedPoints[i].y);
                if (shiftedPoints[i].x != 0) {
                    angles[i] = (Math.atan(shiftedPoints[i].y / shiftedPoints[i].x) * 180 / Math.PI) - angleNose;
                } else {
                    angles[i] = 90 - angleNose;
                }
            }
        }

    }

    public static void main(String[] args) {

        svm_parameter param = new svm_parameter();

        Vector<FeatureSet> featureSetVector = new Vector<>();

        // Read from files all folders, their expressions.

        File sortedSet = new File("sorted_set");
        String landmarkData;

        for (File directory : sortedSet.listFiles()) {
            for (File textFile : directory.listFiles()) {
                try {
                    landmarkData = new String(Files.readAllBytes(textFile.toPath()));
                    featureSetVector.add(new FeatureSet(landmarkData, directory.getName()));
                } catch (IOException ioe) {
                    ioe.printStackTrace();
                }
            }
        }

        // default values
        param.svm_type = svm_parameter.C_SVC;
        param.kernel_type = svm_parameter.RBF;
        param.degree = 3;
        param.gamma = 0;
        param.coef0 = 0;
        param.nu = 0.5;
        param.cache_size = 40;
        param.C = 1;
        param.eps = 1e-3;
        param.p = 0.1;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = new int[0];
        param.weight = new double[0];

        // build problem
        svm_problem prob = new svm_problem();
        prob.l = featureSetVector.size();
        prob.y = new double[prob.l];
        prob.x = new svm_node[prob.l][272];
        for (int i = 0; i < prob.l ; i++) {
            FeatureSet featureSet = featureSetVector.elementAt(i);
            for (int j = 0; j < 68; j++) {
                int k = j * 4;
                prob.x[i][k] = new svm_node();
                prob.x[i][k].index = k + 1;
                prob.x[i][k].value = featureSet.shiftedPoints[j].x;
                k++;
                prob.x[i][k] = new svm_node();
                prob.x[i][k].index = k + 1;
                prob.x[i][k].value = featureSet.shiftedPoints[j].y;
                k++;
                prob.x[i][k] = new svm_node();
                prob.x[i][k].index = k + 1;
                prob.x[i][k].value = featureSet.distances[j];
                k++;
                prob.x[i][k] = new svm_node();
                prob.x[i][k].index = k + 1;
                prob.x[i][k].value = featureSet.angles[j];
            }
            prob.y[i] = featureSet.expression;
        }

        // build model & classify
        svm_model model = svm.svm_train(prob, param);

        try {
            svm.svm_save_model("svm_data.dat", model);
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

}
