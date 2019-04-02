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
package fit.astro.vsa.common.datahandling.training.multi;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class TestMultiViewData {
    
    private final Map<Integer, List<RealVector>> setOfTestingPatterns;
    private final Map<Integer, String> setOfTestingClasses;

    /**
     * 
     * @param setOfPatterns
     * @param setOfClasses
     * @param testingData 
     */
    public TestMultiViewData(
            Map<Integer, List<RealVector>> setOfPatterns,
            Map<Integer, String> setOfClasses,
            List<Integer> testingData) {
        this.setOfTestingPatterns = new HashMap<>();
        this.setOfTestingClasses = new HashMap<>();
        
        testingData.stream().map((idx) -> {
            setOfTestingPatterns.put(idx, setOfPatterns.get(idx));
            return idx;
        }).forEach((idx) -> {
            setOfTestingClasses.put(idx, setOfClasses.get(idx));
        });

    }

    /**
     * @return the setOfTestingPatterns
     */
    public Map<Integer, List<RealVector>> getSetOfTestingPatterns() {
        return setOfTestingPatterns;
    }

    /**
     * @return the setOfTestingClasses
     */
    public Map<Integer, String> getSetOfTestingClasses() {
        return setOfTestingClasses;
    }
}

