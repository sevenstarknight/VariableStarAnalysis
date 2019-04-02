/*
 * Copyright (C) 2018 kjohnston
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
package fit.astro.vsa.analysis.ucr;

import fit.astro.vsa.common.utilities.math.handling.exceptions.NotEnoughDataException;
import java.io.IOException;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author kjohnston
 */
public class UCR_Test {

    public UCR_Test() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

//        double[] tauArray = new double[]{0.25, 1.0, 1.75};
//        double[] muArray = new double[]{2.0, 5.0, 8.0};
//        double[] lambdaArray = new double[]{0.1, 0.5, 1.0};
    
//        @Test
    public void testUCR() throws IOException, NotEnoughDataException {

        String fileLocation = "/Users/kjohnston/Google Drive/VarStarData/UCR";

        ProcessUCRDataViaL3ML.execute(0.25, 5.0, 0.5, false, fileLocation);

    }
    
    
    
    @Test
    public void testUCR_MV() throws IOException, NotEnoughDataException {

        String fileLocation = "/Users/kjohnston/Google Drive/VarStarData/UCR";

        ProcessUCRDataViaL3ML_MV.execute(0.5, 0.0, 0.5, true, fileLocation);

    }
}
