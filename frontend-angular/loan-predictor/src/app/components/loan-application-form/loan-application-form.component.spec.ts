import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LoanApplicationFormComponent } from './loan-application-form.component';

describe('LoanApplicationFormComponent', () => {
  let component: LoanApplicationFormComponent;
  let fixture: ComponentFixture<LoanApplicationFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [LoanApplicationFormComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(LoanApplicationFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
