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
package fit.astro.vsa.common.utilities.io;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import fit.astro.vsa.common.bindings.ml.TimeDomainAttributeMaps;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author kjohnston
 */
public class GenerateExampleFile {
    
    private static final Logger LOGGER
            = LoggerFactory.getLogger(GenerateExampleFile.class);
    
    public static void main(String[] args){
        
        
        ReadingInUCRData starLight = new ReadingInUCRData("UCR");

        TimeDomainAttributeMaps trainingData = starLight.getTrainData("StarLightCurves");
        TimeDomainAttributeMaps testingData = starLight.getTestData("StarLightCurves");
        
        System.out.println("training: " + trainingData.getSetOfClasses().size());
        System.out.println("testing: " + testingData.getSetOfClasses().size());
        
        // ===============
        
        double[][] errorMLWith = new double[2][2];
        MLArray tmpArray = new MLDouble("tmp", errorMLWith);
        
        List<MLArray> list = new ArrayList<>();
        list.add(tmpArray);

        MatlabFunctions.storeToFinal("Error-" + "UCR.mat", list);
    }
}
