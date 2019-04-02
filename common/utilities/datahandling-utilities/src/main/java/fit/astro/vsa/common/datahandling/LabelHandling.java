/*
 * Copyright (C) 2016 Kyle Johnston
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package fit.astro.vsa.common.datahandling;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class LabelHandling {

    /**
     *
     * @param setOfPatterns
     * @param setOfClasses
     * <p>
     * @return
     */
    public static Map<String, List<List<RealVector>>> sortIntoListClasses(
            Map<Integer, List<RealVector>> setOfPatterns,
            Map<Integer, String> setOfClasses) {
        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, List<List<RealVector>>> classMembers = new HashMap<>();
        setOfPatterns.keySet().stream().forEach((currentPatternId) -> {
            String classType = setOfClasses.get(currentPatternId);
            List<RealVector> pattern = setOfPatterns.get(currentPatternId);

            if (classMembers.containsKey(classType)) {
                List<List<RealVector>> listOfPatterns
                        = classMembers.get(classType);

                listOfPatterns.add(pattern);
                classMembers.put(classType, listOfPatterns);
            } else {
                List<List<RealVector>> listOfPatterns = new ArrayList<>();
                listOfPatterns.add(pattern);
                classMembers.put(classType, listOfPatterns);
            }
        });

        return classMembers;
    }

    /**
     *
     * @param setOfPatterns
     * @param setOfClasses
     * <p>
     * @return
     */
    public static Map<String, List<RealVector>> sortIntoClasses(
            Map<Integer, RealVector> setOfPatterns,
            Map<Integer, String> setOfClasses) {
        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, List<RealVector>> classMembers = new HashMap<>();
        setOfPatterns.keySet().stream().forEach((currentPatternId) -> {
            String classType = setOfClasses.get(currentPatternId);
            RealVector pattern = setOfPatterns.get(currentPatternId);

            if (classMembers.containsKey(classType)) {
                List<RealVector> listOfPatterns
                        = classMembers.get(classType);

                listOfPatterns.add(pattern);
                classMembers.put(classType, listOfPatterns);
            } else {
                List<RealVector> listOfPatterns = new ArrayList<>();
                listOfPatterns.add(pattern);
                classMembers.put(classType, listOfPatterns);
            }
        });

        return classMembers;
    }

    /**
     *
     * @param setOfPatterns
     * @param setOfClasses
     * <p>
     * @return
     */
    public static Map<String, List<RealMatrix>> sortIntoMatrixClasses(
            Map<Integer, RealMatrix> setOfPatterns,
            Map<Integer, String> setOfClasses) {
        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, List<RealMatrix>> classMembers = new HashMap<>();
        setOfPatterns.keySet().stream().forEach((currentPatternId) -> {
            String classType = setOfClasses.get(currentPatternId);
            RealMatrix pattern = setOfPatterns.get(currentPatternId);

            if (classMembers.containsKey(classType)) {
                List<RealMatrix> listOfPatterns
                        = classMembers.get(classType);

                listOfPatterns.add(pattern);
                classMembers.put(classType, listOfPatterns);
            } else {
                List<RealMatrix> listOfPatterns = new ArrayList<>();
                listOfPatterns.add(pattern);
                classMembers.put(classType, listOfPatterns);
            }
        });

        return classMembers;
    }

    /**
     *
     * @param setOfPatterns
     * @param setOfClasses
     * <p>
     * @return
     */
    public static Map<String, Map<Integer, RealVector>> sortIntoMaps(
            Map<Integer, RealVector> setOfPatterns,
            Map<Integer, String> setOfClasses) {
        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, Map<Integer, RealVector>> classMembers = new HashMap<>();
        setOfPatterns.keySet().stream().forEach((currentPatternId) -> {
            String classType = setOfClasses.get(currentPatternId);
            RealVector pattern = setOfPatterns.get(currentPatternId);

            if (classMembers.containsKey(classType)) {
                Map<Integer, RealVector> mapOfPatterns
                        = classMembers.get(classType);

                mapOfPatterns.put(currentPatternId, pattern);
                classMembers.put(classType, mapOfPatterns);
            } else {
                Map<Integer, RealVector> listOfPatterns = new HashMap<>();
                listOfPatterns.put(currentPatternId, pattern);
                classMembers.put(classType, listOfPatterns);
            }
        });

        return classMembers;
    }

    /**
     *
     * @param mapOfClasses
     * <p>
     * @return
     */
    public static Map<String, List<Integer>> sortIntoMaps(
            Map<Integer, String> mapOfClasses) {
        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, List<Integer>> classMembers = new HashMap<>();
        mapOfClasses.keySet().stream().forEach((currentPatternId) -> {
            String classType = mapOfClasses.get(currentPatternId);

            if (classMembers.containsKey(classType)) {
                List<Integer> listOfIntegers
                        = classMembers.get(classType);

                listOfIntegers.add(currentPatternId);
                classMembers.put(classType, listOfIntegers);
            } else {
                List<Integer> listOfIntegers = new ArrayList<>();
                listOfIntegers.add(currentPatternId);
                classMembers.put(classType, listOfIntegers);
            }
        });

        return classMembers;
    }

    /**
     *
     * @param setOfPatterns
     * @param setOfClasses
     * <p>
     * @return
     */
    public static Map<String, Map<Integer, RealMatrix>> sortIntoMatrixMaps(
            Map<Integer, RealMatrix> setOfPatterns,
            Map<Integer, String> setOfClasses) {
        // ========================================
        // Set up: Split Input Patterns Into Class Groups
        Map<String, Map<Integer, RealMatrix>> classMembers = new HashMap<>();
        setOfPatterns.keySet().stream().forEach((currentPatternId) -> {
            String classType = setOfClasses.get(currentPatternId);
            RealMatrix pattern = setOfPatterns.get(currentPatternId);

            if (classMembers.containsKey(classType)) {
                Map<Integer, RealMatrix> mapOfPatterns
                        = classMembers.get(classType);

                mapOfPatterns.put(currentPatternId, pattern);
                classMembers.put(classType, mapOfPatterns);
            } else {
                Map<Integer, RealMatrix> listOfPatterns = new HashMap<>();
                listOfPatterns.put(currentPatternId, pattern);
                classMembers.put(classType, listOfPatterns);
            }
        });

        return classMembers;
    }

    /**
     *
     * @param setOfClasses
     * <p>
     * @return
     */
    public static Map<String, Integer> countUniqueClasses(
            Map<Integer, String> setOfClasses) {
        Map<String, Integer> uniqueClasses = new HashMap<>();

        setOfClasses.keySet().stream().forEach((trainingDataID) -> {
            if (uniqueClasses.containsKey(
                    setOfClasses.get(trainingDataID))) {
                String classLabel = setOfClasses.get(trainingDataID);
                Integer currentCount = uniqueClasses.get(classLabel);
                uniqueClasses.put(classLabel, currentCount + 1);
            } else {
                String classLabel = setOfClasses.get(trainingDataID);
                uniqueClasses.put(classLabel, 1);
            }
        });

        return uniqueClasses;

    }

}
