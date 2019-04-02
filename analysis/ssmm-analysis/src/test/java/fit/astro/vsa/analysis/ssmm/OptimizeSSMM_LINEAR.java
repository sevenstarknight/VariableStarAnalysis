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
package fit.astro.vsa.analysis.ssmm;

import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import fit.astro.vsa.common.bindings.analysis.astro.VarStarDataset;
import fit.astro.vsa.common.bindings.math.Real2DCurve;
import fit.astro.vsa.common.utilities.io.MatlabFunctions;
import fit.astro.vsa.common.utilities.io.dataset.ReadingInLINEARData;
import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import fit.astro.vsa.common.datahandling.training.TrainCrossTestGenerator;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author Kyle Johnston kyjohnst2000@my.fit.edu
 */
public class OptimizeSSMM_LINEAR {

    public static void main(String[] args) throws IOException, NotEnoughDataException {

        String fileLocation = "/Users/kjohnston/Desktop/AstroData/Linear/LinearData.mat";

        ReadingInLINEARData lINEARData = new ReadingInLINEARData(fileLocation);

        VarStarDataset dataset = lINEARData.getLinearDataset();

        Map<Integer, String> setOfClasses = dataset.getMultiViewSideData().getSetOfClasses();
                
        
        TrainCrossTestGenerator crossTestGenerator = new TrainCrossTestGenerator(setOfClasses,
                0.5, new Random(42L));

        Map<Integer, String> setOfClassesTraining = new HashMap();
        Map<Integer, Real2DCurve> tdDataTraining = new HashMap();

        for (Integer idx : crossTestGenerator.getTrainingData()) {
            setOfClassesTraining.put(idx, setOfClasses.get(idx));
            tdDataTraining.put(idx, dataset.getTimeD().get(idx));
        }
        
        // ===============================================================
        SSMMAnalysis analysis = new SSMMAnalysis(
                tdDataTraining,
                setOfClassesTraining);
        
        List<RealVector> errorArray = analysis.optimizeSSMM_Dependent();

        RealMatrix errorMetric = new Array2DRowRealMatrix(
                errorArray.get(0).getDimension(),
                errorArray.size());

        for(int idx = 0; idx < errorArray.size(); idx++){
            errorMetric.setColumnVector(idx, errorArray.get(idx));
        }
        
        MLArray errorML = new MLDouble("error", errorMetric.getData());
        List<MLArray> list = new ArrayList<>();
        list.add(errorML);
        
        MatlabFunctions.storeToTestAnalysis("SSMM-Error-LINEAR.mat", list);
    }
}
