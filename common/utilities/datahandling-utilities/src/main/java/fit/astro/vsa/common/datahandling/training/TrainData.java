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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class TrainData {
    
    private final Map<Integer, RealVector> setOfTrainingPatterns;
    private final Map<Integer, String> setOfTrainingClasses;

    /**
     * 
     * @param setOfPatterns
     * @param setOfClasses
     * @param trainingData 
     */
    public TrainData(
            Map<Integer, RealVector> setOfPatterns,
            Map<Integer, String> setOfClasses,
            List<Integer> trainingData) {
        this.setOfTrainingPatterns = new HashMap<>();
        this.setOfTrainingClasses = new HashMap<>();
        
        trainingData.stream().map((idx) -> {
            setOfTrainingPatterns.put(idx, setOfPatterns.get(idx));
            return idx;
        }).forEach((idx) -> {
            setOfTrainingClasses.put(idx, setOfClasses.get(idx));
        });

    }

    /**
     * @return the setOfTestingPatterns
     */
    public Map<Integer, RealVector> getSetOfTrainingPatterns() {
        return setOfTrainingPatterns;
    }

    /**
     * @return the setOfTestingClasses
     */
    public Map<Integer, String> getSetOfTrainingClasses() {
        return setOfTrainingClasses;
    }
}

