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
package fit.astro.vsa.common.datahandling.training;

import fit.astro.vsa.common.datahandling.LabelHandling;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class TrainCrossGenerator {

    private final Map<String, List<Integer>> classMembers;

    private final List<Integer> trainingData;
    private final Map<Integer, List<Integer>> crossvalMap;

    /**
     * 5-fold cross val
     *
     * @param setOfClasses
     * @param rand
     */
    public TrainCrossGenerator(
            Map<Integer, String> setOfClasses, Random rand) {

        this.classMembers
                = LabelHandling.sortIntoMaps(setOfClasses);

        //=========================================================
        // Sort into training data
        trainingData = new ArrayList<>();

        for (String label : classMembers.keySet()) {

            List<Integer> tempList = classMembers.get(label);

            tempList.stream().forEach((idx) -> {
                trainingData.add(idx);
            });
        }

        // Randomly Permutate the TrainingSet
        Collections.shuffle(trainingData, rand);

        //=========================================================
        // 5-fold cross val
        crossvalMap = new HashMap<>();

        int size = trainingData.size() / 5;
        Iterator<Integer> iterTraining = trainingData.iterator();
        for (int i = 0; i < 5; i++) {
            List<Integer> idxSet = new ArrayList<>();

            int counter = 0;
            while (iterTraining.hasNext()) {
                idxSet.add(iterTraining.next());
                counter++;
                if (counter > size) {
                    break;
                }
            }
            crossvalMap.put(i, idxSet);
        }

    }

    /**
     *
     * @param setOfClasses
     * @param segments
     * @return
     */
    public static Map<Integer, List<Integer>> generateCrossVal123(
            Map<Integer, String> setOfClasses, int segments) {

        Set<Integer> setOfIdx = setOfClasses.keySet();

        Map<Integer, List<Integer>> crossValSet = new HashMap<>();
        for (int idx = 0; idx < segments; idx++) {
            crossValSet.put(idx, new ArrayList<>());
        }

        int counter = 0;
        for (Integer idx : setOfIdx) {
            crossValSet.get(counter % segments).add(idx);
            counter++;
        }

        return crossValSet;
    }

    public Map<String, List<Integer>> getClassMembers() {
        return classMembers;
    }

    /**
     * @return the crossvalMap
     */
    public Map<Integer, List<Integer>> getCrossvalMap() {
        return crossvalMap;
    }

    /**
     * @return the trainingData
     */
    public List<Integer> getTrainingData() {
        return trainingData;
    }
}
