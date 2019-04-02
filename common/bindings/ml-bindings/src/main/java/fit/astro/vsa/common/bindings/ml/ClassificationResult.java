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
package fit.astro.vsa.common.bindings.ml;

import java.util.Map;

/**
 *
 * @author Kyle Johnston <kyjohnst2000@my.fit.edu>
 */
public class ClassificationResult {

    // <editor-fold defaultstate="collapsed" desc="Variables">
    
    private final Map<Integer, String> labelEstimate;
    private final Map<Integer, Map<String, Double>> labelAndPostProb;
    private final Map<String, Integer> trainingDataCount;
    private Double threshold;

    // </editor-fold>
    
    /**
     *
     * @param labelEstimate input pattern -> key; label -> value
     * @param labelAndPostProb input patterns -> key ; label/prob map -> value
     * @param uniqueLabelCount -> labels
     */
    public ClassificationResult(
            Map<Integer, String> labelEstimate,
            Map<Integer, Map<String, Double>> labelAndPostProb,
            Map<String, Integer> uniqueLabelCount) {
        this.labelEstimate = labelEstimate;
        this.labelAndPostProb = labelAndPostProb;
        this.trainingDataCount = uniqueLabelCount;
    }

    // <editor-fold defaultstate="collapsed" desc="Getters/Setters">
    /**
     * 
     * @return 
     */
    public Double getThreshold() {
        return threshold;
    }

    /**
     * 
     * @param threshold 
     */
    public void setThreshold(Double threshold) {
        this.threshold = threshold;
    }

    /**
     * 
     * @return 
     */
    public Map<Integer, String> getLabelEstimate() {
        return labelEstimate;
    }

    /**
     * 
     * @return 
     */
    public Map<Integer, Map<String, Double>> getLabelAndPostProb() {
        return labelAndPostProb;
    }

    /**
     * 
     * @return 
     */
    public Map<String, Integer> getTrainingDataCount() {
        return trainingDataCount;
    }
// </editor-fold>
}
