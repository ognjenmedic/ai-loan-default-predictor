import { Component, ElementRef, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { environment } from '../../environments/environment';

@Component({
  selector: 'app-loan-application-form',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './loan-application-form.component.html',
  styleUrls: ['./loan-application-form.component.css'],
})
export class LoanApplicationFormComponent {
  private apiUrl = environment.apiUrl;
  dummyPayload: any = null;
  formData = {
    OCCUPATION_TYPE: '',
    AMT_INCOME_TOTAL: 0,
    AMT_CREDIT: 0,
    loanTermMonths: 0, // map from pos_cash_balance_data[0].CNT_INSTALMENT
    EXT_SOURCE_2: 0,
  };

  dtiWarning: string = '';

  predictionResult: any = null;

  occupations: string[] = [
    'Laborers',
    'Core staff',
    'Accountants',
    'Managers',
    'Drivers',
    'Sales staff',
    'Cleaning staff',
    'Cooking staff',
    'Private service staff',
    'Medicine staff',
    'Security staff',
    'High skill tech staff',
    'Waiters/barmen staff',
    'Low-skill Laborers',
    'Realty agents',
    'Secretaries',
    'IT staff',
    'HR staff',
  ];

  @ViewChild('predictionOutput') predictionResultSection!: ElementRef;

  constructor(private http: HttpClient) {}

  generateDummyData() {
    this.http
      .get<any>(`${this.apiUrl}/generate_dummy`)
      .subscribe((response) => {
        this.dummyPayload = response;

        const monthlyIncome = response.application.AMT_INCOME_TOTAL ?? 0;
        let loanAmount = response.application.AMT_CREDIT ?? 0;
        const loanTermMonths =
          response.pos_cash_balance_data?.[0]?.CNT_INSTALMENT ?? 0;

        // Prevent DTI from being too high
        if (monthlyIncome > 0 && loanTermMonths > 0) {
          const maxLoanAmount = monthlyIncome * loanTermMonths * 0.3; // Max loan ensuring DTI ≤ 30%
          if (loanAmount > maxLoanAmount) {
            loanAmount = Math.floor(maxLoanAmount); // Adjust loan amount to be valid
          }
        }

        // Populate the form with the adjusted values
        this.formData = {
          OCCUPATION_TYPE: response.application.OCCUPATION_TYPE ?? '',
          AMT_INCOME_TOTAL: monthlyIncome,
          AMT_CREDIT: loanAmount,
          loanTermMonths: loanTermMonths,
          EXT_SOURCE_2: response.application.EXT_SOURCE_2 ?? 0,
        };

        this.checkDTI(); // Ensure the UI updates immediately
      });
  }

  predictLoan() {
    if (!this.dummyPayload) {
      console.error('No dummy payload available. Generate first.');
      return;
    }

    const RUBLE_TO_USD = 90;
    const rubIncome = Math.round(this.formData.AMT_INCOME_TOTAL * RUBLE_TO_USD);
    const rubCredit = Math.round(this.formData.AMT_CREDIT * RUBLE_TO_USD);

    this.dummyPayload.application.OCCUPATION_TYPE =
      this.formData.OCCUPATION_TYPE;
    this.dummyPayload.application.AMT_INCOME_TOTAL = rubIncome;
    this.dummyPayload.application.AMT_CREDIT = rubCredit;

    if (this.dummyPayload.pos_cash_balance_data?.[0]) {
      this.dummyPayload.pos_cash_balance_data[0].CNT_INSTALMENT =
        this.formData.loanTermMonths;
    }
    this.dummyPayload.application.EXT_SOURCE_2 = this.formData.EXT_SOURCE_2;

    console.log('[DEBUG] Final payload to /predict:', this.dummyPayload);

    this.http.post<any>(`${this.apiUrl}/predict`, this.dummyPayload).subscribe(
      (response) => {
        this.predictionResult = response;

        // Wait for Angular to update the UI, then scroll to result
        setTimeout(() => {
          this.scrollToResult();
        }, 100);
      },
      (error) => {
        console.error('Error predicting loan:', error);
      }
    );
  }

  scrollToResult() {
    if (this.predictionResultSection) {
      this.predictionResultSection.nativeElement.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    }
  }

  getProbabilityNumber(): number {
    return parseFloat(this.predictionResult?.formatted_probability ?? '0');
  }

  updateNumber(event: any, field: keyof typeof this.formData) {
    let value = event.target.value.replace(/,/g, ''); // Remove existing commas
    if (!isNaN(value) && field in this.formData) {
      (this.formData as any)[field] = Number(value);
    }

    // Check if we need to show a DTI warning
    this.checkDTI();
  }

  checkDTI() {
    const monthlyIncome = this.formData.AMT_INCOME_TOTAL;
    const loanAmount = this.formData.AMT_CREDIT;
    const loanTermMonths = this.formData.loanTermMonths;

    if (monthlyIncome > 0 && loanAmount > 0 && loanTermMonths > 0) {
      const estimatedAnnuity = loanAmount / loanTermMonths; // Approximate monthly payment
      const dtiRatio = estimatedAnnuity / monthlyIncome; // Debt-to-Income ratio

      if (dtiRatio > 0.3) {
        this.dtiWarning = `DTI ratio too high (${(dtiRatio * 100).toFixed(
          1
        )}%). Please choose a lower loan amount.`;
      } else {
        this.dtiWarning = ''; // Clear the warning if ratio is acceptable
      }
    }
  }

  isDTITooHigh(): boolean {
    const monthlyIncome = this.formData.AMT_INCOME_TOTAL;
    const loanAmount = this.formData.AMT_CREDIT;
    const loanTermMonths = this.formData.loanTermMonths;

    if (monthlyIncome > 0 && loanAmount > 0 && loanTermMonths > 0) {
      const estimatedAnnuity = loanAmount / loanTermMonths; // Approximate monthly payment
      const dtiRatio = estimatedAnnuity / monthlyIncome; // Debt-to-Income ratio

      return dtiRatio > 0.3; // Disable button if DTI is greater than 30%
    }
    return false; // Button remains enabled
  }
}
